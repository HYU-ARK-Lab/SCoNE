# 실험 방법들 모아놓은 코드
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import os
from collections import Counter
from tqdm import tqdm
import pandas as pd

class NeuronStrategies:
    """
    실험들 모아놓은 클래스
    ircan (experiment_mode='ircan'). counter_attribution_desc(attr_results) — IRCAN baseline
    scone (+ scone_attronly/varonly/var/std/mad/w1/w2/w5/w10). high_attr_high_var_counter_enhance /
        high_attr_high_dispersion_counter_enhance / high_attr_high_var_counter_enhance_window — SCoNE (ours)
    exp_cross. cross_dataset_intersection_enhance(train_dataset_name, cross_ds_name, top_k)
    """
    
    """
    all_neuron_stats = {
        (0, 100): [0.00001, 0.00002, 0.000015, ...],  # 30개 데이터셋의 점수
        (0, 101): [0.00003, 0.00002, 0.000018, ...],
        (1, 50): [0.00005, 0.00004, 0.000052, ...],
        ...
    }
    """
    @staticmethod
    def _pos_list2str(pos_list):
        return '@'.join([str(pos) for pos in pos_list])

    @staticmethod
    def _pos_str2list(pos_str):
        return [int(pos) for pos in pos_str.split('@')]
    
    @classmethod
    def cross_dataset_intersection_enhance(cls, *args, neuron_json=None, **kwargs):
        """
        exp_cross.
        hotpotqa ∩ 2wiki scone top30 교집합 뉴런 9개를 바로 반환.
        neuron_json이 주어지면 그 final_selection.json에서 cns를 로드해 반환 (마이닝 스킵).
        """
        if neuron_json is not None:
            return cls.load_cns_from_json(neuron_json)
        # llama-3-8b        
        # ircan
        # cns = [
        #     [27, 8140],
        #     [13, 2158],
        #     [14, 12683],
        #     [19, 10739],
        #     [30, 5035],
        # ]

        # ircan top15
        # cns = [
        #     [27, 8140],
        #     [13, 2158],
        #     [14, 12683],
        #     [19, 10739],
        #     [30, 5035],
        #     [30, 3382],
        #     [31, 154],
        #     [21, 12666],
        #     [14, 6477],
        #     [31, 1386],
        #     [0, 12829],
        #     [14, 11805],
        #     [14, 5370],
        #     [31, 11929],
        #     [31, 1730],
        # ]

        
        # scone_w3
        # hotpotqa_neuron 5개
        cns = [
            [27, 8140],
            [13, 2158],
            [21, 12666],
            [30, 3382],
            [30, 5035]
        ]

        # scone_w1
        # cns = [
        #     [13, 2158],
        #     [27, 8140],
        #     [30, 3382],
        #     [21, 12666],
        #     [31, 154],
        # ]

        # scone_w10
        # cns = [
        #     [13, 2158],
        #     [27, 8140],
        #     [30, 5035],
        #     [31, 154],
        #     [21, 12666],
        # ]
        
        # hotpotqa_neuron scone 500개로 뽑은 거
        # cns = [
        #     [27, 8140],
        #     [13, 2158],
        #     [14, 12683],
        #     [19, 10739],
        #     [30, 3382],
        # ]

        # hotpotqa_neuron scone 1000개로 뽑은 거
        # cns = [
        #     [27, 8140],
        #     [13, 2158],
        #     [14, 12683],
        #     [19, 10739],
        #     [21, 12666],
        # ]
        
        # hotpotqa_neuron 15개
        # cns = [
        #     [27, 8140],
        #     [13, 2158],
        #     [21, 12666],
        #     [30, 3382],
        #     [30, 5035],
        #     [0, 12829],
        #     [31, 154],
        #     [19, 10739],
        #     [31, 1386],
        #     [14, 12683],
        #     [27, 10277],
        #     [31, 5115],
        #     [31, 10549],
        #     [31, 11929],
        #     [31, 13725]
        # ]

        # qwen. hotpotqa. scone으로 뽑은 뉴런
        # cns = [
        #     [11, 616],
        #     [27, 12776],
        #     [27, 9228],
        #     [18, 17881],
        #     [27, 9577],
        # ]

        # qwen. hotpotqa. ircan으로 뽑은 뉴런
        # cns = [
        #     [11, 616],
        #     [27, 9228],
        #     [27, 12776],
        #     [27, 11288],
        #     [8, 5988],
        # ]

        # cns = NeuronStrategies.load_cns_from_json("/home/irteam/chaewon-workspace/bergen/neuron_analysis_results/hotpotqa_n100_random_seed11_meta-llama_Meta-Llama-3-8B-Instruct_scone_top5/final_selection.json")
        print(f"[exp_cross] 교집합 뉴런 {len(cns)}개 사용")
        return cns

    @staticmethod
    def load_cns_from_json(path):
        """final_selection.json 파일을 읽어서 cns = [[layer, neuron], ...] 형태로 반환."""
        with open(path, 'r') as f:
            data = json.load(f)
        cns = [[n['layer'], n['neuron']] for n in data['selected_neurons']]
        print(f"[load_cns_from_json] {len(cns)}개 뉴런 로드: {path}")
        return cns

    @classmethod
    def counter_attribution_desc(cls, attr_results, top_k, train_dataset_name=None, model_name=None):
        """
        ircan.
        기존 방식: Counter로 가장 많이 등장한 뉴런 세기"""
        enhance_cn_num = top_k
        
        # Context Neuron 상위 20개 추출한다.
        cn_bag_list = []
        for i, res_dict in enumerate(tqdm(attr_results, desc="Extracting Context Neurons (Original)")):
            try:
                metric_triplets = res_dict["all_attr_gold"]
                metric_triplets.sort(key=lambda x: x[2], reverse=True) # 74, 73개의 뉴런들 중 중요도 높은 순으로 정렬한다음에
                cn_bag = metric_triplets[:20]
                cn_bag_list.append(cn_bag)
            except Exception as e:
                print(f"Error extracting CNS for sample {i}: {e}")
                continue

        # CNS를 enhance한 모델로 결과를 만든다.
        cns = []
        cn_counter = Counter()
        for cn_bag in cn_bag_list:  
            for cn in cn_bag:
                cn_counter.update([cls._pos_list2str(cn[:2])])

        most_common_cn = cn_counter.most_common(enhance_cn_num)
        cns = [cls._pos_str2list(cn_str[0]) for cn_str in most_common_cn]

        output_dir = Path("neuron_analysis_results")
        output_dir.mkdir(exist_ok=True)

        safe_model = model_name.replace("/", "_") if model_name else None
        prefix = f"{train_dataset_name}_{safe_model}_ircan_top{enhance_cn_num}" if safe_model else f"{train_dataset_name}_ircan_top{enhance_cn_num}"

        counter_data = {
            'model_name': model_name,
            'top_k': enhance_cn_num,
            'selected_neurons': [
                {
                    'neuron': cn_str,
                    'layer': cls._pos_str2list(cn_str)[0],
                    'neuron_id': cls._pos_str2list(cn_str)[1],
                    'count': count
                }
                for cn_str, count in most_common_cn
            ]
        }
        with open(output_dir / f"{prefix}_final_selection.json", "w") as f:
            json.dump(counter_data, f, indent=2)
        print(f"[Saved] {output_dir / f'{prefix}_final_selection.json'}")

        return cns

    @classmethod
    def high_attr_high_var_counter_enhance(cls, attr_results, all_neuron_stats, top_k, train_dataset_name, model_name=None, top_n=50, select_mode='intersection'):
        
        all_var = {}
        for (layer, neuron), scores in all_neuron_stats.items():
            dataset_size = len(scores)
            curr_attr_scores = np.array(scores)

            step_var = []
            # Variability 계산
            for i in range(3, dataset_size):
                prev_mean = curr_attr_scores[i-3:i].mean()
                #1. 
                var = abs(curr_attr_scores[i] - prev_mean)

                step_var.append(var)
        
            all_var[(layer, neuron)] = step_var
        
        #2. High Variability를 토대로 dataset마다 top 50개 저장
        #2-1. High Attr을 토대로 dataset마다 top 50개 저장
        #2-2. Intersection (layer, neuron)을 저장
        cn_bag_list = []
        dataset_size = len(next(iter(all_neuron_stats.values())))
        intersection_log = [] 

        for t in range(dataset_size - 3):
            high_var_key = []
            high_attr_key = []
            
            for (layer, neuron), var_list in all_var.items():
                
                current_attr_score = all_neuron_stats[(layer, neuron)][t + 3]
                if (current_attr_score > 0):
                    high_var_key.append({
                        'layer': layer,
                        'neuron': neuron,
                        'var': var_list[t]
                    })
            
            for (layer, neuron), scores in all_neuron_stats.items():
                
                current_attr_score = all_neuron_stats[(layer, neuron)][t + 3]
                if (current_attr_score > 0):
                    high_attr_key.append({
                        'layer': layer,
                        'neuron': neuron,
                        'attr': scores[t + 3]
                    })

            high_var_key.sort(key=lambda x: x['var'], reverse=True)
            high_attr_key.sort(key=lambda x: x['attr'], reverse=True)

            top_50_high_var = high_var_key[:top_n]
            top_50_high_attr = high_attr_key[:top_n]

            high_var_set = {(n['layer'], n['neuron']) for n in top_50_high_var}
            high_attr_set = {(n['layer'], n['neuron']) for n in top_50_high_attr}
            if select_mode == 'attr_only':      # attr@top_n control (variability 제거)
                intersection = high_attr_set
            elif select_mode == 'var_only':     # variability-only control
                intersection = high_var_set
            else:                                # 'intersection' = SCoNE (default, 기존과 동일)
                intersection = high_var_set & high_attr_set

            print(f"INTERSECTION 개수: {len(intersection)}")


            cn_bag = []
            # 현재 데이터셋의 점수를 누적 통계에 추가
            for layer, neuron in intersection:
                cn_bag.append([layer, neuron])
            
            cn_bag_list.append(cn_bag)
            
            intersection_log.append({
            'dataset_index': t,
            'intersection_count': len(intersection),
            'neurons': [[layer, neuron] for layer, neuron in intersection]
            })
        
        # Counter: 얼마나 자주 등장했는지
        cn_counter = Counter()
        for cn_bag in cn_bag_list:
            for cn in cn_bag:
                cn_counter.update([cls._pos_list2str(cn)])

        most_common = cn_counter.most_common(top_k)
        cns = [cls._pos_str2list(neuron_str) for neuron_str, _ in most_common]

        safe_model_name = model_name.replace("/", "_") if model_name else None
        model_suffix = f"_{safe_model_name}" if safe_model_name else ""
        mode_tag = {'intersection': 'scone', 'attr_only': 'scone_attronly', 'var_only': 'scone_varonly'}[select_mode]
        prefix = f"{train_dataset_name}{model_suffix}_{mode_tag}_top{top_k}_n{top_n}"

        output_dir = Path("neuron_analysis_results") / prefix
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "intersection_log.json", "w") as f:
            json.dump(intersection_log, f, indent=2)

        # 2. Counter 결과 (모든 뉴런의 등장 횟수)
        counter_data = {
            'model_name': model_name,
            'total_datasets': dataset_size - 3,
            'all_neurons': [
                {
                    'neuron': neuron_str,
                    'layer': cls._pos_str2list(neuron_str)[0],
                    'neuron_id': cls._pos_str2list(neuron_str)[1],
                    'count': count,
                    'frequency': count / (dataset_size - 3)
                }
                for neuron_str, count in cn_counter.most_common()
            ]
        }

        with open(output_dir / "counter_results.json", "w") as f:
            json.dump(counter_data, f, indent=2)

        # 3. Top K 최종 선택된 뉴런
        final_selection = {
            'model_name': model_name,
            'top_k': top_k,
            'selected_neurons': [
                {
                    'layer': cn[0],
                    'neuron': cn[1],
                    'count': cn_counter[cls._pos_list2str(cn)],
                    'frequency': cn_counter[cls._pos_list2str(cn)] / (dataset_size - 3)
                }
                for cn in cns
            ]
        }

        with open(output_dir / "final_selection.json", "w") as f:
            json.dump(final_selection, f, indent=2)

        # 4. 요약 통계 (텍스트 파일)
        with open(output_dir / f"{prefix}_summary.txt", "w") as f:
            f.write("=== Neuron Selection Summary ===\n\n")
            f.write(f"Total datasets analyzed: {dataset_size - 3}\n")
            f.write(f"Total unique neurons in counter: {len(cn_counter)}\n")
            f.write(f"Top K selected: {top_k}\n\n")

            f.write("=== Intersection Statistics ===\n")
            for log in intersection_log:
                f.write(f"Dataset {log['dataset_index']}: {log['intersection_count']} neurons\n")

            f.write(f"\nAverage intersection size: {np.mean([log['intersection_count'] for log in intersection_log]):.2f}\n\n")

            f.write("=== Top 10 Most Frequent Neurons ===\n")
            for i, (cn, count) in enumerate(most_common[:10], 1):
                layer, neuron = cls._pos_str2list(cn)
                f.write(f"{i}. Layer {layer}, Neuron {neuron}: {count} times ({count/(dataset_size-3)*100:.1f}%)\n")

        return cns

    @classmethod
    def high_attr_high_dispersion_counter_enhance(cls, attr_results, all_neuron_stats, top_k, train_dataset_name, model_name=None, top_n=50, select_mode='intersection', var_mode='var'):
        # scone(high_attr_high_var_counter_enhance)의 복사본.
        # variability를 running-residual(|Attr_i - mean(직전 3개)|, 순서 의존) 대신
        # 뉴런당 전역(순서 불변) dispersion 스칼라로 정의한다 (rebuttal 대안 정의 비교용).
        #   μ(n) = (1/N) Σ_t Attr_t(n)   ← 전체 N개 입력 평균
        #   'var' : (1/N) Σ_t (Attr_t - μ)^2      (L2, 제곱)
        #   'std' : sqrt(var)                     (L2, 원래 단위; ranking은 var와 동일)
        #   'mad' : (1/N) Σ_t |Attr_t - μ|         (L1, 이상치 강건)
        # 스칼라 1개를 전 step에 broadcast → 매 t마다 선택이 동일 (순서 완전 불변).
        all_var = {}
        for (layer, neuron), scores in all_neuron_stats.items():
            dataset_size = len(scores)
            curr_attr_scores = np.array(scores)

            mu = curr_attr_scores.mean()            # μ(n)
            dev = curr_attr_scores - mu
            if var_mode == 'var':
                var_scalar = float(np.mean(dev ** 2))
            elif var_mode == 'std':
                var_scalar = float(np.sqrt(np.mean(dev ** 2)))
            elif var_mode == 'mad':
                var_scalar = float(np.mean(np.abs(dev)))
            else:
                raise ValueError(f"Unknown var_mode: {var_mode} (var/std/mad 중 하나)")

            step_var = [var_scalar] * (dataset_size - 3)   # broadcast (원본 step 개수와 동일하게 유지)
            all_var[(layer, neuron)] = step_var
            

        cn_bag_list = []
        dataset_size = len(next(iter(all_neuron_stats.values())))
        intersection_log = []

        for t in range(dataset_size - 3):
            high_var_key = []
            high_attr_key = []

            for (layer, neuron), var_list in all_var.items():
                current_attr_score = all_neuron_stats[(layer, neuron)][t + 3]
                if (current_attr_score > 0):
                    high_var_key.append({
                        'layer': layer,
                        'neuron': neuron,
                        'var': var_list[t]
                    })

            for (layer, neuron), scores in all_neuron_stats.items():
                current_attr_score = all_neuron_stats[(layer, neuron)][t + 3]
                if (current_attr_score > 0):
                    high_attr_key.append({
                        'layer': layer,
                        'neuron': neuron,
                        'attr': scores[t + 3]
                    })

            high_var_key.sort(key=lambda x: x['var'], reverse=True)
            high_attr_key.sort(key=lambda x: x['attr'], reverse=True)

            top_50_high_var = high_var_key[:top_n]
            top_50_high_attr = high_attr_key[:top_n]

            high_var_set = {(n['layer'], n['neuron']) for n in top_50_high_var}
            high_attr_set = {(n['layer'], n['neuron']) for n in top_50_high_attr}
            if select_mode == 'attr_only':      # attr@top_n control (variability 제거)
                intersection = high_attr_set
            elif select_mode == 'var_only':     # variability-only control
                intersection = high_var_set
            else:                                # 'intersection' = SCoNE (default)
                intersection = high_var_set & high_attr_set

            print(f"INTERSECTION 개수: {len(intersection)}")

            cn_bag = []
            for layer, neuron in intersection:
                cn_bag.append([layer, neuron])

            cn_bag_list.append(cn_bag)

            intersection_log.append({
            'dataset_index': t,
            'intersection_count': len(intersection),
            'neurons': [[layer, neuron] for layer, neuron in intersection]
            })

        cn_counter = Counter()
        for cn_bag in cn_bag_list:
            for cn in cn_bag:
                cn_counter.update([cls._pos_list2str(cn)])

        most_common = cn_counter.most_common(top_k)
        cns = [cls._pos_str2list(neuron_str) for neuron_str, _ in most_common]

        safe_model_name = model_name.replace("/", "_") if model_name else None
        model_suffix = f"_{safe_model_name}" if safe_model_name else ""
        base_tag = {'intersection': 'sconedisp', 'attr_only': 'sconedisp_attronly', 'var_only': 'sconedisp_varonly'}[select_mode]
        mode_tag = f"{base_tag}_{var_mode}"     # 예: sconedisp_var / sconedisp_std / sconedisp_mad
        prefix = f"{train_dataset_name}{model_suffix}_{mode_tag}_top{top_k}_n{top_n}"

        output_dir = Path("neuron_analysis_results") / prefix
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "intersection_log.json", "w") as f:
            json.dump(intersection_log, f, indent=2)

        counter_data = {
            'model_name': model_name,
            'total_datasets': dataset_size - 3,
            'all_neurons': [
                {
                    'neuron': neuron_str,
                    'layer': cls._pos_str2list(neuron_str)[0],
                    'neuron_id': cls._pos_str2list(neuron_str)[1],
                    'count': count,
                    'frequency': count / (dataset_size - 3)
                }
                for neuron_str, count in cn_counter.most_common()
            ]
        }

        with open(output_dir / "counter_results.json", "w") as f:
            json.dump(counter_data, f, indent=2)

        final_selection = {
            'model_name': model_name,
            'top_k': top_k,
            'selected_neurons': [
                {
                    'layer': cn[0],
                    'neuron': cn[1],
                    'count': cn_counter[cls._pos_list2str(cn)],
                    'frequency': cn_counter[cls._pos_list2str(cn)] / (dataset_size - 3)
                }
                for cn in cns
            ]
        }

        with open(output_dir / "final_selection.json", "w") as f:
            json.dump(final_selection, f, indent=2)

        with open(output_dir / f"{prefix}_summary.txt", "w") as f:
            f.write("=== Neuron Selection Summary (dispersion) ===\n\n")
            f.write(f"Variability mode: {var_mode} (global, order-invariant)\n")
            f.write(f"Total datasets analyzed: {dataset_size - 3}\n")
            f.write(f"Total unique neurons in counter: {len(cn_counter)}\n")
            f.write(f"Top K selected: {top_k}\n\n")

            f.write("=== Intersection Statistics ===\n")
            for log in intersection_log:
                f.write(f"Dataset {log['dataset_index']}: {log['intersection_count']} neurons\n")

            f.write(f"\nAverage intersection size: {np.mean([log['intersection_count'] for log in intersection_log]):.2f}\n\n")

            f.write("=== Top 10 Most Frequent Neurons ===\n")
            for i, (cn, count) in enumerate(most_common[:10], 1):
                layer, neuron = cls._pos_str2list(cn)
                f.write(f"{i}. Layer {layer}, Neuron {neuron}: {count} times ({count/(dataset_size-3)*100:.1f}%)\n")

        return cns

    @classmethod
    def high_attr_high_var_counter_enhance_window(cls, attr_results, all_neuron_stats, top_k, train_dataset_name, model_name=None, window_size=3, top_n=50):
        all_var = {}
        for (layer, neuron), scores in all_neuron_stats.items():
            dataset_size = len(scores)
            curr_attr_scores = np.array(scores)

            step_var = []
            for i in range(window_size, dataset_size):
                prev_mean = curr_attr_scores[i-window_size:i].mean()
                var = abs(curr_attr_scores[i] - prev_mean)
                step_var.append(var)

            all_var[(layer, neuron)] = step_var

        cn_bag_list = []
        dataset_size = len(next(iter(all_neuron_stats.values())))
        intersection_log = []

        for t in range(dataset_size - window_size):
            high_var_key = []
            high_attr_key = []

            for (layer, neuron), var_list in all_var.items():
                current_attr_score = all_neuron_stats[(layer, neuron)][t + window_size]
                if (current_attr_score > 0):
                    high_var_key.append({
                        'layer': layer,
                        'neuron': neuron,
                        'var': var_list[t]
                    })

            for (layer, neuron), scores in all_neuron_stats.items():
                current_attr_score = all_neuron_stats[(layer, neuron)][t + window_size]
                if (current_attr_score > 0):
                    high_attr_key.append({
                        'layer': layer,
                        'neuron': neuron,
                        'attr': scores[t + window_size]
                    })

            high_var_key.sort(key=lambda x: x['var'], reverse=True)
            high_attr_key.sort(key=lambda x: x['attr'], reverse=True)

            top_50_high_var = high_var_key[:top_n]
            top_50_high_attr = high_attr_key[:top_n]

            high_var_set = {(n['layer'], n['neuron']) for n in top_50_high_var}
            high_attr_set = {(n['layer'], n['neuron']) for n in top_50_high_attr}
            intersection = high_var_set & high_attr_set

            print(f"INTERSECTION 개수: {len(intersection)}")

            cn_bag = []
            for layer, neuron in intersection:
                cn_bag.append([layer, neuron])

            cn_bag_list.append(cn_bag)

            intersection_log.append({
                'dataset_index': t,
                'intersection_count': len(intersection),
                'neurons': [[layer, neuron] for layer, neuron in intersection]
            })

        cn_counter = Counter()
        for cn_bag in cn_bag_list:
            for cn in cn_bag:
                cn_counter.update([cls._pos_list2str(cn)])

        most_common = cn_counter.most_common(top_k)
        cns = [cls._pos_str2list(neuron_str) for neuron_str, _ in most_common]

        safe_model_name = model_name.replace("/", "_") if model_name else None
        model_suffix = f"_{safe_model_name}" if safe_model_name else ""
        prefix = f"{train_dataset_name}{model_suffix}_scone_w{window_size}_top{top_k}_n{top_n}"

        output_dir = Path("neuron_analysis_results") / prefix
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "intersection_log.json", "w") as f:
            json.dump(intersection_log, f, indent=2)

        total = dataset_size - window_size
        counter_data = {
            'model_name': model_name,
            'total_datasets': total,
            'all_neurons': [
                {
                    'neuron': neuron_str,
                    'layer': cls._pos_str2list(neuron_str)[0],
                    'neuron_id': cls._pos_str2list(neuron_str)[1],
                    'count': count,
                    'frequency': count / total
                }
                for neuron_str, count in cn_counter.most_common()
            ]
        }

        with open(output_dir / "counter_results.json", "w") as f:
            json.dump(counter_data, f, indent=2)

        final_selection = {
            'model_name': model_name,
            'top_k': top_k,
            'selected_neurons': [
                {
                    'layer': cn[0],
                    'neuron': cn[1],
                    'count': cn_counter[cls._pos_list2str(cn)],
                    'frequency': cn_counter[cls._pos_list2str(cn)] / total
                }
                for cn in cns
            ]
        }

        with open(output_dir / "final_selection.json", "w") as f:
            json.dump(final_selection, f, indent=2)

        with open(output_dir / f"{prefix}_summary.txt", "w") as f:
            f.write("=== Neuron Selection Summary ===\n\n")
            f.write(f"Window size: {window_size}\n")
            f.write(f"Total datasets analyzed: {total}\n")
            f.write(f"Total unique neurons in counter: {len(cn_counter)}\n")
            f.write(f"Top K selected: {top_k}\n\n")

            f.write("=== Intersection Statistics ===\n")
            for log in intersection_log:
                f.write(f"Dataset {log['dataset_index']}: {log['intersection_count']} neurons\n")

            f.write(f"\nAverage intersection size: {np.mean([log['intersection_count'] for log in intersection_log]):.2f}\n\n")

            f.write("=== Top 10 Most Frequent Neurons ===\n")
            for i, (cn, count) in enumerate(most_common[:10], 1):
                layer, neuron = cls._pos_str2list(cn)
                f.write(f"{i}. Layer {layer}, Neuron {neuron}: {count} times ({count/total*100:.1f}%)\n")

        return cns

