import os
import numpy as np
import matplotlib.pyplot as plt

def get_ratio(aps, clfs, target):
    emap, dmap, emapx, dmap = aps[0][1], aps[1][1], aps[2][1], aps[3][1]
    candidate_n = len(emap)
    max_r = 0
    for i in range(candidate_n):
        pcls = clfs[i]
        pwe, pwh, pse, psh = emap[i][1], dmap[i][1], emapx[i][1], dmap[i][1]
        A = pcls * (pwe - psh) + (1 - pcls) * (pse - pwh)
        B = (1 - pcls) * pwh + pcls * psh
        r = (target - B) / A
        r_star = min(r, 1)
        max_r = max(max_r, r_star)
    return max_r

def dynamic_ap(aps, clfs, r):
    emap, dmap, emapx, dmapx = aps[0][1], aps[1][1], aps[2][1], aps[3][1]
    candidate_n = len(emap)
    # dynamic_ap列表
    dynamic_aps = []
    r_index = 0
    r = [item/100 for item in r]
    # 将 r 转换为集合以提高查询效率
    r_set = set(r) if isinstance(r, list) else r
    for i in range(candidate_n):
        if emap[i][0] not in r_set:
            continue
        pcls = clfs[r_index]
        pwe, pwh, pse, psh = emap[i][1], dmap[i][1], emapx[i][1], dmapx[i][1]
        p_correct_e = r[i] * pcls * pwe
        p_wrong_h = (1 - r[i]) * (1 - pcls) * pwh
        p_correct_h = (1 - r[i]) * pcls * psh
        p_wrong_e = r[i] * (1 - pcls) * pse
        ptotal = p_correct_e + p_wrong_h + p_correct_h + p_wrong_e
        dynamic_aps.append(ptotal)
        r_index += 1
    return dynamic_aps

def resolve_npz(npz_file):
    fs = np.load(npz_file)
    result = []
    for key in fs.files:
        result.append((key, fs[key]))
    return result

class TestDiscrimnn:
    def test_getratio(self):
        expected_ap = 0.95
            # 解析npz文件，用户获取ap的迭代曲线
        npz_path = os.path.join('/home/insslab/Desktop/Vi-DiscrimNN/exp/data', 'val_iter_map_pestv3.npz')
        iter_aps = resolve_npz(npz_path)
        length = len(iter_aps[0][1])
        clfs = np.ones(length)
        ratio = get_ratio(iter_aps, clfs, expected_ap)
        print(ratio)
    def test_dynamic_ap(self):
        npz_path = os.path.join('/home/insslab/Desktop/Vi-DiscrimNN/exp/data', 'val_iter_map_pestv3.npz')
        iter_aps = resolve_npz(npz_path)
        length = len(iter_aps[0][1])
        clfs = np.ones(length)
        r = np.arange(10, 100, 5).tolist()
        dynamic_aps = dynamic_ap(iter_aps, clfs, r)
        plt.figure()
        plt.plot(range(length), dynamic_aps)
        plt.ylim((0.8, 0.975))
        plt.savefig('qwq.png')
        print(dynamic_aps)
tgt = TestDiscrimnn()
tgt.test_dynamic_ap()