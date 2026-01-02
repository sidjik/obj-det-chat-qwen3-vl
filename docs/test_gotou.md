# Gotou Picture

We used this image as a test to ensure the model sees the image at all, so we'll include some usage examples. You can see the original image below.

![gotou test image](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_gotou/gotou.jpg)


### Question: 1

- *Question*: what are you see on this image
- *Output*: model provide only text output as exptected

#### Chat-example
![chat describe image](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_gotou/chat_1.png)



### Question: 2

- *Question*: where exactly locates name of the anime title on the image
- *Output*: expected only bounding object generation, but model provide answer + bounding object

#### Chat-example
![chat name image 1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_gotou/chat_2_1.png)
![chat name image 2](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_gotou/chat_2_2.png)


