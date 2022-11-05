def purity_score(y, y_pre):
    """purity_score(y,y_pre)

        Computes the purity score.

        @arg y: np_array ground truth

        @arg y_pre: result of the sklearn clusterization

        @return: value of the clusterization's purity

    """
    m = {}
    for i in range(len(y)):  # traverse the results matrix and store the objects that belong to each cluster on a map
        if y_pre[i] in m:
            m[y_pre[i]].append(i)
        else:
            m[y_pre[i]] = [i]

    tot = 0

    for i in m:  # check the most common class in each of the clusters
        c = {}
        mx = 0  # number of members of the most common class in the cluster
        for j in m[i]:
            if y[j] in c:
                c[y[j]] += 1
            else:
                c[y[j]] = 1
            mx = max(mx, c[y[j]])
        tot += mx
    return tot / len(y)


def collocation_score(y, y_pre):
    """collocation_score(y,y_pre)

        Computes the colocation score.

        @arg y: np_array ground truth

        @arg y_pre: result of the sklearn clusterization

        @return: value of the clusterization's colocation

    """
    return purity_score(y_pre, y)  # to calculate the colocation measure just calculate the purity in reverse way


def harmonicMean_score(y=[], y_pre=[], p=0, c=0):
    """harmonicMean_score(y=0,y_pre=0,p=0,c=0)

        Computes the Harmonic Mean score.

        @arg y: np_array ground truth

        @arg y_pre: result of the sklearn clusterization

        or

        @arg p: pre computed purity value

        @arg c: pre computed collocation value

        @return: value of the clusterization's harmonic mean

    """
    if p == c == 0:
        p = purity_score(y, y_pre)
        c = collocation_score(y, y_pre)
    return (2 * c * p) / (c + p)
