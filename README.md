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



