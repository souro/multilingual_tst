# Performance Analysis of LLMs for Text Style Transfer Tasks in English, Hindi, and Bengali

This repo contains the code and data of the papers: [Multilingual Text Style Transfer: Datasets & Models for Indian Languages](https://aclanthology.org/2024.inlg-main.41/) and [Low-Resource Text Style Transfer for Bangla: Data & Models](https://aclanthology.org/2023.banglalp-1.5/)

## Overview
This repository focuses on Text Style Transfer (TST), specifically sentiment transfer, across nine Indian languages: Bengali, Hindi, Magahi, Malayalam, Marathi, Punjabi, Odia, Telugu, and Urdu (we refined and corrected an existing English dataset, and we introduce new human-translated multilingual datasets that parallels its English counterpart.). We introduce dedicated datasets containing 1,000 positive and 1,000 negative style-parallel sentences for each language. Our evaluation covers various benchmark models, including Llama2 and GPT-3.5, categorized into parallel, non-parallel, cross-lingual, and shared learning approaches. Our experiments emphasize the importance of parallel data in TST and the effectiveness of the Masked Style Filling (MSF) technique. Additionally, cross-lingual and joint multilingual learning methods show promise, providing insights for selecting optimal models for specific language and task requirements. This work is the first comprehensive study of TST as sentiment transfer across such a diverse range of languages.

## Data
You can find the data and all the necessary details [here](https://github.com/panlingua/multilingual-tst-datasets).

## Walkthrough
*Will add more information in this section soon.*

### Dependency

    pip install -r requirements.txt

## Citing
If you use this data or code please cite the following:
  
    @inproceedings{mukherjee-etal-2024-multilingual-text,
    title = "Multilingual Text Style Transfer: Datasets {\&} Models for {I}ndian Languages",
    author = "Mukherjee, Sourabrata  and
      Ojha, Atul Kr.  and
      Bansal, Akanksha  and
      Alok, Deepak  and
      McCrae, John P.  and
      Dusek, Ondrej",
    editor = "Mahamood, Saad  and
      Minh, Nguyen Le  and
      Ippolito, Daphne",
    booktitle = "Proceedings of the 17th International Natural Language Generation Conference",
    month = sep,
    year = "2024",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.inlg-main.41",
    pages = "494--522",
    abstract = "",
}
}

## License

    Author: Sourabrata Mukherjee
    Copyright © 2023 Sourabrata Mukherjee.
    Licensed under the MIT License.

## Acknowledgements

This research was funded by the European Union (ERC, NG-NLG, 101039303) and Charles University project SVV 260 698. We acknowledge the use of resources provided by the LINDAT/CLARIAH-CZ Research Infrastructure (Czech Ministry of Education, Youth, and Sports project No. LM2018101). We also acknowledge Panlingua Language Processing LLP for collaborating on this research project. Atul Kr. Ojha would like to acknowledge the support of the Science Foundation Ireland (SFI) as part of Grant Number SFI/12/RC/2289_P2 Insight_2, Insight SFI Research Centre for Data Analytics.
