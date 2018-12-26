from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.tree import tree

if __name__ == '__main__':
    with open('lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 特征标签
    lenses_list = []  # 保存lenses数据的临时列表
    lenses_dict = {}  # 保存lenses数据的字典，用于生成pandas
    print(lenses_target)
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])  # make column to list
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    print()
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)

    clf = tree.DecisionTreeClassifier(max_depth=4)
    print(lenses_pd.values.tolist())
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    print(clf.predict([[1, 1, 1, 0]]))
