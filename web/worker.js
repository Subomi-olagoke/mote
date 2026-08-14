/* web/worker.js — the engine lives here, off the main thread.
 *
 * The worker fetches the model and tokenizer straight into WASM memory (no
 * second copy in a JS ArrayBuffer), stands the engine up through the same
 * mote_lib API the iOS app links against, and streams generated text back to
 * the page. Generation blocks this worker, which is fine: the page stays live.
 */
importScripts('mote.js');

let Module = null;
let handle = 0;
let isBPE = false;
const decoder = new TextDecoder('utf-8');   // stream:true reassembles split UTF-8

/* Fetch a URL directly into the WASM heap. Returns [ptr, length]. */
async function fetchIntoHeap(url, label) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const total = Number(res.headers.get('Content-Length')) || 0;

    if (!total) {                            // no length header: buffer, then copy
        const buf = new Uint8Array(await res.arrayBuffer());
        const ptr = Module._malloc(buf.length);
        Module.HEAPU8.set(buf, ptr);
        return [ptr, buf.length];
    }

    const ptr = Module._malloc(total);
    let off = 0;
    const reader = res.body.getReader();
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        Module.HEAPU8.set(value, ptr + off);  // re-read HEAPU8: growth remaps it
        off += value.length;
        postMessage({ type: 'progress', label, loaded: off, total });
    }
    return [ptr, off];
}

async function boot(modelURL, tokURL) {
    Module = await createMote();

    const [tokPtr, tokLen] = await fetchIntoHeap(tokURL, 'tokenizer');
    const [modPtr, modLen] = await fetchIntoHeap(modelURL, 'model');
    isBPE = Module.HEAPU8[tokPtr] === 0x4D && Module.HEAPU8[tokPtr + 1] === 0x54; // "MT"

    handle = Module._mote_create(modPtr, modLen, tokPtr, tokLen);
    if (!handle) throw new Error('mote_create failed');

    /* mote_info_t: 9 consecutive ints */
    const infoPtr = Module._malloc(9 * 4);
    Module.ccall('mote_get_info', null, ['number', 'number'], [handle, infoPtr]);
    const i = new Int32Array(Module.HEAP32.buffer, infoPtr, 9);
    const info = { dim: i[0], hidden: i[1], layers: i[2], heads: i[3], kvHeads: i[4],
                   vocab: i[5], ctx: i[6], quantized: i[7], qbits: i[8] };
    Module._free(infoPtr);
    postMessage({ type: 'ready', info });
}

function generate(userText, maxNew, system, history) {
    /* Instruct models (BPE tokenizer) get the ChatML template, with the page's
     * system prompt and recent turns replayed for context; the TinyStories
     * model is a continuation model and takes the text raw. */
    let prompt = userText;
    if (isBPE) {
        prompt = `<|im_start|>system\n${system}<|im_end|>\n`;
        for (const t of history || [])
            prompt += `<|im_start|>user\n${t.user}<|im_end|>\n` +
                      `<|im_start|>assistant\n${t.assistant}<|im_end|>\n`;
        prompt += `<|im_start|>user\n${userText}<|im_end|>\n<|im_start|>assistant\n`;
    }

    /* mote_params: { f32 temperature, f32 topp, f32 repeat_penalty,
     *                (4 pad), u64 seed, i32 max_tokens }  = 32 bytes */
    const pPtr = Module._malloc(32);
    Module.HEAPF32[pPtr >> 2] = 0.0;          // greedy
    Module.HEAPF32[(pPtr >> 2) + 1] = 0.9;
    Module.HEAPF32[(pPtr >> 2) + 2] = 1.15;   // repeat penalty: stops verbatim loops
    Module.HEAP32[(pPtr >> 2) + 4] = 0;       // seed (unused when greedy)
    Module.HEAP32[(pPtr >> 2) + 5] = 0;
    Module.HEAP32[(pPtr >> 2) + 6] = maxNew | 0;

    const t0 = performance.now();
    let tokens = 0, tFirst = 0;
    const cb = Module.addFunction((piecePtr) => {
        let end = piecePtr;
        while (Module.HEAPU8[end] !== 0) end++;
        const text = decoder.decode(Module.HEAPU8.subarray(piecePtr, end), { stream: true });
        if (++tokens === 1) tFirst = performance.now();
        if (text) postMessage({ type: 'token', text });
        return 1;
    }, 'iii');

    Module.ccall('mote_generate', 'number',
                 ['number', 'string', 'number', 'number', 'number'],
                 [handle, prompt, pPtr, cb, 0]);

    const t1 = performance.now();
    const tail = decoder.decode();            // flush any buffered partial UTF-8
    if (tail) postMessage({ type: 'token', text: tail });
    postMessage({ type: 'done', tokens,
                  msToFirst: tFirst ? tFirst - t0 : 0,
                  seconds: (t1 - t0) / 1000,
                  tokensPerSecond: tokens > 1 ? (tokens - 1) / ((t1 - tFirst) / 1000) : 0 });
    Module.removeFunction(cb);
    Module._free(pPtr);
}

onmessage = (e) => {
    const m = e.data;
    if (m.type === 'boot')
        boot(m.model, m.tokenizer).catch(err => postMessage({ type: 'error', message: String(err) }));
    else if (m.type === 'generate')
        try { generate(m.text, m.maxNew || 200, m.system || 'You are a helpful assistant.', m.history); }
        catch (err) { postMessage({ type: 'error', message: String(err) }); }
};
