In this repository, I implemented a model inspired in the paper Large-Scale Long-Tailed Recognition in an Open World (OLTR).

Table I. Obtained experimental results in the MNIST-ML.
| Approach                 | Acc.  |
|--------------------------|-------|
| CLS                      | 84.28 |
| CLS + OLTR 		  | 92.13 |


In each folder (ex. exp_cls), I provide the configuration (config.json) employed to achieve these results. To train a model using a specific configuration:

```python
# create a novel environment and then
pip install -r requeriments.txt
train.py exp_cls\config.json
```
More details coming soon ...


References:

[Large-Scale Long-Tailed Recognition in an Open World](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjN_prg7_r0AhXIF7kGHTw6DCQQFnoECAYQAQ&url=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent_CVPR_2019%2Fpapers%2FLiu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf&usg=AOvVaw1pTCShb7rN2C7v0MkaaZbE)


