# TODO:
1. Integrate pretokenized nanogpt dataset. Fineweb
2. Integrate muon for 2d layers and 8bit adam for other params
3. Track all layer's attention block and ffn block output in wandb
4. Add eigen decomposition or spectral insights for the individual layers and weight. Run it at the end of every 100/200 th iteration depending on speed of doing it.
5. tqdm instead of print. Also at every 10 th step. Because .item is expensive.
6. Integrate flash attention
7. Integrate other triton kernels if possible.
