# Saliency Prediction on Mobile Videos: A Large-scale Dataset and A Transformer Approach
![dataset](https://github.com/wenshijie110/MVFormer/assets/54231028/24281504-fb64-42d3-bda7-1c39b6fd1058)

This is the official code of paper 'Saliency Prediction on Mobile Videos: A Large-scale Dataset and A Transformer Approach' (will be updated once our paper is accepted)

With the booming development of smart devices, mobile videos have drawn broad interest when humans surf social media. Different from traditional long-form videos, mobile videos are featured with uncertain human attention behavior so far owing to the specific displaying mode, thus promoting the research on saliency prediction for mobile videos. Unfortunately, the current eye-tracking experiments are not applicable for mobile videos, since the stationary eye-tracker and eye fixation acquisition are dedicated to the videos presented on computers. To tackle this issue, we propose performing the wearable eye-tracker to record viewers' egocentric fixations and then devising a fixation mapping technique to project the eye fixations from egocentric videos onto mobile videos. Resorting to this technique, the large-scale mobile video saliency (MVS) dataset is established, including 1,007 mobile videos and 5,935,927 fixations. Given this dataset, we exhaustively analyze the characteristics of subjects' fixations and two findings are obtained. Based on the MVS dataset and these findings, we propose a saliency prediction approach on mobile videos upon Video Swin Transformer (MVFormer), wherein long-range spatio-temporal dependency is captured to derive the human attention mechanism on mobile videos. In MVFormer, we develop the selective feature fusion module to balance multi-scale features, and the progressive saliency prediction module to generate saliency maps via progressive aggregation of multi-scale features. Extensive experiments have shown that our MVFormer approach significantly outperforms other state-of-the-art saliency prediction approaches.

### Visualization of predicted saliency maps of our and other compared approaches over two mobile video clips in MVS dataset.

![visualization](https://github.com/wenshijie110/MVFormer/assets/54231028/6ade5405-8148-4bfa-a7f9-2d66fb35e6fd)



