# Efficient and Accurate Referring Image Segmentation

Final Report - https://www.overleaf.com/project/6758b1ffde51eee9e8b58f39

Base model - ETRIS source code: https://github.com/kkakkkka/ETRIS

Running instructions are the same as for ETRIS and stated on its github page. There are some special instructions for LoRA and Cross-Modal Fusion extensions that are specified in respective branches.

Branches and extensions:
- **main** - ```Task Specific Decoder``` by <ins>```Sanga Park```</ins>
  - Linformer for Efficient Attention - To optimize the computational efficiency of the attention mechanism, we replaced the existing \texttt{nn.MultiheadAttention} module in the HA module with Linformer. Dynamic Position-Aware Encoding - Instead of the fixed position encoding provided by CoordConv in GA  module, we implemented a learnable positional encoding mechanism based on Dynamic Positional Encoding.
- **evgenii** - ```LoRA``` by <ins>```Evgenii Venediktov```</ins>
  - Special LoRA.MultiheadAttention replaces regular MultiheadAttention in model.clip.py, LoRA rank set to model_d//2. Different modes of training (regular / only LoRA) configured from yaml config file.
- **selective-feature-adaptation** - ```Selective Feature Adaptation``` by <ins>```Uma Sharma```</ins>
  - the SelectiveAdaptationLayer replaces InteractorT in layers.py. SelectiveAdaptationLayer is used instead of InteractorT in the HA l(Heirarchical Alignment) layers. Run details are the same as ETRIS main code.
- **hyunwoo** - ```Cross-Modal Fusion``` by <ins>```Hyunwoo Yu```</ins>
  - The proposed method is designed with a structure that receives multi-modal features in the encoder and fuses target-informative information. This is implemented in lib/decoder.py.
