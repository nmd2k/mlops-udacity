# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model utilizes the `RandomForestClassifier` class from the `sklearn.model` module to perform classification tasks.

The parameters are set to default.

## Intended Use

The pretrained model use for predict a salary (over or under 50k) based on the census data.

## Training Data

Features:

- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

For both training and evaluation, categorical features of the data are encoded using OneHotEncoder and the target is transformed using LabelBinarizer

## Evaluation Data

The training and evaluation data are splitted from the original set with 80:20 ratio.

## Metrics
Precision: 0.7364457831325302
Recall: 0.6197718631178707
Fbeta: 0.6730901582931865

## Ethical Considerations

This model is trained on census data. Since the dataset contains sensitive attributes like `race`, `sex`, etc. special care must be taken to avoid any potential biases or unfair treatment of individuals or groups.

## Caveats and Recommendations

I recommend that checks are included upstream of any decision-making points to ensure that bias is minimized.