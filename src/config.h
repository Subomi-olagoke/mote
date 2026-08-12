/* config.h — the shape of a model.
 *
 * Seven integers are all it takes to describe a Llama-style transformer's
 * dimensions. Everything else in the engine is derived from these.
 */
#ifndef MOTE_CONFIG_H
#define MOTE_CONFIG_H

typedef struct {
    int dim;         /* residual-stream width                                */
    int hidden_dim;  /* inner width of the feed-forward block                */
    int n_layers;    /* number of transformer blocks                         */
    int n_heads;     /* attention query heads                                */
    int n_kv_heads;  /* key/value heads (< n_heads means grouped-query attn) */
    int vocab_size;  /* token count                                          */
    int seq_len;     /* max context length the weights were trained for      */
} Config;

#endif /* MOTE_CONFIG_H */
