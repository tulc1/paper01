## WWW'22 HRCF: Enhancing Collaborative Filtering via Hyperbolic Geometric Regularization


Authors: Menglin Yang, Min Zhou, Jiahong Liu, Defu Lian, Irwin King

Note: this repository is built upon [HGCF](https://github.com/layer6ai-labs/HGCF) and [HGCN](https://github.com/HazyResearch/hgcn). By the way, if you would like to list HRCF as a baseline, please follow the parameter's setting, such as the layers of HRCF is 8 for slightly dense dataset.
 
<a name="Environment"/>

## Environment:
The code was developed and tested on the following python environment:
```
python 3.7.7
pytorch 1.5.1
scikit-learn 0.23.2
numpy 1.20.2
scipy 1.6.2
tqdm 4.60.0
```
<a name="instructions"/>

## Instructions:

Train and evaluation HRCF:

- To evaluate HRCF on Amazon_CD 
  - `bash ./example/run_cd.sh`
- To evaluate HRCF on Amazon_Book
   - `bash ./example/run_book.sh`
- To evaluate HRCF on Yelp
    - `bash ./example/run_yelp.sh`

<a name="citation"/>

## Notes

During our research, we came across LGCF's works. However, when attempting to reproduce their findings, we found that LGCF conducted experiments using different settings with the baseline, i.e., training for 1000 epochs compared to 500 training epochs used by HGCF and other baselines. When we reduced LGCF's training epochs to 500, it barely showed improvement. On the other hand, extending the training epochs of HGCF and HRCF to 1000 led to improvement correspondingly. This illustrates that fully hyperbolic models may not necessarily have a substantial impact on user-item representation. For a fail and consistent comparison, we did not list the result of LGCF.

## Citation

If you find this code useful in your research, please cite the following paper:

    @inproceedings{yang2022hrcf,
      title={HRCF: Enhancing Collaborative Filtering via Hyperbolic Geometric Regularization},
      author={Menglin Yang, Min Zhou, Jiahong Liu, Defu Lian, Irwin King},
      booktitle={Proceedings of the International World Wide Web Conference},
      year={2022}
    }


