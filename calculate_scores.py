"""
score the result of the LLMs
"""

import os
import re
import json
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from utils import TEST_TYPE_MAP

class Scoring:
    def __init__(self, res_dir, score_dir, data_dir):
        """initialize the scoring class
        Args:
            res_dir (str): the path to the result of the LLMs
            passing_path (str): the path to the csv of metrics for the passing score of tests
            score_dir (str): the path to save the scoring result
            data_dir (str): the path to the ground truth data
        """
        self.res_dir = res_dir
        self.score_dir = score_dir
        self.data_dir = data_dir

    # Helper function to normalize answers
    def normalize_answer(self, answer):
        # Remove non-alphabetic non-numeric characters, sort the characters, and convert to uppercase. the length of the answer is limited to <6
        return ''.join(sorted(re.sub(r'[^a-zA-Z0-9]', '', answer))).upper()[:6]

    def ishi_score(self, year, answer_res_path, save_path, fix_format=False):
        """score the result of the LLMs on 医師国試
        Args:
            year (str): the year of the exam
            answer_res_path (str): the path to the answer of the LLM
            save_path (str): the path to save the scoring result
            fix_format (bool): if True, fix the format of the answer of the problems
        Returns:
            total_score (int[]): the total score of the LLM in the exam, including 必修（B,E） and 一般（A,C,D,F）
            pass_or_not (bool): whether the LLM passed the exam
            failed_by_forbidden (bool): if over 3 forbidden choices were selected, return True
        """

        # 必修: B、E；一般: A、C、D、F
        must_sections = ["B", "E"]
        general_sections = ["A", "C", "D", "F"]

        pass_score_dict = {
            "2020": [158, 217], # 必修158, 一般217
            "2021": [160, 209], # 必修160, 一般209
            "2022": [158, 214], # 必修158, 一般214
            "2023": [160, 220], # 必修160, 一般220
            "2024": [160, 230]  # 必修160, 一般230
        }

        total_score = [0, 0] # [必修、一般]
        count_forbidden = 0
        pass_or_not = False
        failed_by_forbidden = False

        # read the ground truth data (correct answer, points, forbidden choices, etc.)
        ground_truth_path = os.path.join(self.data_dir, "医師")

        for section in ["B", "E", "A", "C", "D", "F"]:
            score_index = 0 if section in must_sections else 1 # 0: 必修, 1: 一般
            # read the ground truth data
            with open(os.path.join(ground_truth_path, f"医師_{year}_{section.lower()}.json"), "r") as f:
                question_data = json.load(f)
            
            # read the answer of the LLM
            with open(os.path.join(answer_res_path, "医師", f"医師_{year}_{section.lower()}_pred.json"), "r") as f:
                answer_data = json.load(f)
            
            history_data = []
            # score the answer of the LLM
            for i, problem in enumerate(question_data):
                answer = answer_data[i]["pred"]
                answer = self.normalize_answer(answer)
                correct_answer = problem["answer"]
                correct_answer = self.normalize_answer(correct_answer)
                if correct_answer != "" and answer == correct_answer:
                    total_score[score_index] += int(problem["points"])
                
                # check if the answer is forbidden
                for choice in answer:
                    if choice in problem["kinki"]:
                        count_forbidden += 1
                        break
                
                history_data.append({
                    "year": year,
                    "section": section,
                    "index": problem["index"],
                    "text_only": problem["text_only"],
                    "kinki": problem["kinki"],
                    "pred": answer,
                    "answer": correct_answer,
                    "points": int(problem["points"]),
                    "human_accuracy": problem["human_accuracy"]
                })
            with open(os.path.join(save_path, "医師", f"医師_{year}_{section.lower()}_history.json"), "w") as f:
                json.dump(history_data, f, indent=4)

        if count_forbidden > 3:
            failed_by_forbidden = True
        
        if total_score[0] >= pass_score_dict[str(year)][0] and total_score[1] >= pass_score_dict[str(year)][1] and not failed_by_forbidden:
            pass_or_not = True
        
        return total_score, pass_or_not, failed_by_forbidden
    
    def shika_score(self, year, answer_res_path, save_path, fix_format=False):
        """score the result of the LLMs on 歯科国試
        Args:
            year (str): the year of the exam
            answer_res_path (str): the path to the answer of the LLM
            save_path (str): the path to save the scoring result
            fix_format (bool): if True, fix the format of the answer of the problems
        Returns:
            total_score (int[]): the total score of the LLM in the exam, including 必修（B,E） and 一般（A,C,D,F）
            pass_or_not (bool): whether the LLM passed the exam
            failed_by_forbidden (bool): if over 3 forbidden choices were selected, return True
        """

        pass_score_dict = {
            "2020": [64, 65, 260], # 必修64, 領域A65, 領域B260
            "2021": [63, 53, 236], # 必修63, 領域A53, 領域B236
            "2022": [64, 59, 237], # 必修64, 領域A59, 領域B237
            "2023": [64, 63, 257], # 必修64, 領域A63, 領域B257
            "2024": [64, 60, 254]  # 必修64, 領域A60, 領域B254
        }

        def get_ryouiki(year, section, index):
            index = int(index.split("-")[0])
            if index <= 20: # 必修
                return 0
            elif index <= 45: # 領域A
                return 1
            else:
                return 2

        total_score = [0, 0, 0] # [必修、領域A、領域B]
        count_forbidden = 0
        pass_or_not = False
        failed_by_forbidden = False

        # read the ground truth data (correct answer, points, forbidden choices, etc.)
        ground_truth_path = os.path.join(self.data_dir, "歯科")

        for section in ["A", "B", "C", "D"]:
            # read the ground truth data
            with open(os.path.join(ground_truth_path, f"歯科_{year}_{section.lower()}.json"), "r") as f:
                question_data = json.load(f)
            
            # read the answer of the LLM
            with open(os.path.join(answer_res_path, "歯科", f"歯科_{year}_{section.lower()}_pred.json"), "r") as f:
                answer_data = json.load(f)
            
            history_data = []
            # score the answer of the LLM
            for i, problem in enumerate(question_data):
                answer = answer_data[i]["pred"]
                answer = self.normalize_answer(answer)
                correct_answer = problem["answer"]
                correct_answer = self.normalize_answer(correct_answer)
                score_index = get_ryouiki(year, section, problem["index"])
                if correct_answer != "" and "corrected_question_index" not in problem and answer == correct_answer:
                    total_score[score_index] += int(problem["points"])
                
                # check if the answer is forbidden
                for choice in answer:
                    if choice in problem["kinki"]:
                        count_forbidden += 1
                        break
                
                history_data.append({
                    "year": year,
                    "section": section,
                    "index": problem["index"],
                    "text_only": problem["text_only"],
                    "kinki": problem["kinki"],
                    "pred": answer,
                    "answer": correct_answer,
                    "points": int(problem["points"]),
                    "human_accuracy": problem["human_accuracy"]
                })
            with open(os.path.join(save_path, "歯科", f"歯科_{year}_{section.lower()}_history.json"), "w") as f:
                json.dump(history_data, f, indent=4)

        if count_forbidden > 3:
            failed_by_forbidden = True
        
        if total_score[0] >= pass_score_dict[str(year)][0] and total_score[1] >= pass_score_dict[str(year)][1] and total_score[2] >= pass_score_dict[str(year)][2] and not failed_by_forbidden:
            pass_or_not = True
        
        return total_score, pass_or_not, failed_by_forbidden

    def kango_score(self, year, answer_res_path, save_path, fix_format=False):
        """score the result of the LLMs on 看護師国試"""
        pass_score_dict = {
            "2020": [40, 155], # 必修40,一般155
            "2021": [40, 159], # 必修40,一般159
            "2022": [40, 167], # 必修40,一般167
            "2023": [40, 152], # 必修40,一般152
            "2024": [40, 158]  # 必修40,一般158
        }

        def get_ryouiki(index):
            index = int(index.split("-")[0])
            if index <= 25: # 必修
                return 0
            else: # 一般
                return 1

        total_score = [0, 0] # [必修、一般]
        pass_or_not = False

        # read the ground truth data (correct answer, points, forbidden choices, etc.)
        ground_truth_path = os.path.join(self.data_dir, "看護")

        for section in ["A", "B"]:
            # read the ground truth data
            with open(os.path.join(ground_truth_path, f"看護_{year}_{section.lower()}.json"), "r") as f:
                question_data = json.load(f)
            
            # read the answer of the LLM
            with open(os.path.join(answer_res_path, "看護", f"看護_{year}_{section.lower()}_pred.json"), "r") as f:
                answer_data = json.load(f)
            
            history_data = []
            # score the answer of the LLM
            for i, problem in enumerate(question_data):
                answer = answer_data[i]["pred"]
                answer = self.normalize_answer(answer)
                correct_answer = problem["answer"]
                correct_answer = self.normalize_answer(correct_answer)
                score_index = get_ryouiki(problem["index"])
                if correct_answer != "" and "corrected_question_index" not in problem and answer == correct_answer:
                    total_score[score_index] += int(problem["points"])
                
                history_data.append({
                    "year": year,
                    "section": section,
                    "index": problem["index"],
                    "text_only": problem["text_only"],
                    "pred": answer,
                    "answer": correct_answer,
                    "points": int(problem["points"]),
                    "human_accuracy": problem["human_accuracy"]
                })
            with open(os.path.join(save_path, "看護", f"看護_{year}_{section.lower()}_history.json"), "w") as f:
                json.dump(history_data, f, indent=4)
        
        if total_score[0] >= pass_score_dict[str(year)][0] and total_score[1] >= pass_score_dict[str(year)][1]:
            pass_or_not = True
        
        return total_score, pass_or_not, False

    def hoken_score(self, year, answer_res_path, save_path, fix_format=False):
        """score the result of the LLMs on 保健師国試"""

        total_score = 0
        pass_or_not = False

        # read the ground truth data (correct answer, points, forbidden choices, etc.)
        ground_truth_path = os.path.join(self.data_dir, "保健")

        for section in ["A", "B"]:
            # read the ground truth data
            with open(os.path.join(ground_truth_path, f"保健_{year}_{section.lower()}.json"), "r") as f:
                question_data = json.load(f)
            
            # read the answer of the LLM
            with open(os.path.join(answer_res_path, "保健", f"保健_{year}_{section.lower()}_pred.json"), "r") as f:
                answer_data = json.load(f)
            
            history_data = []
            # score the answer of the LLM
            for i, problem in enumerate(question_data):
                answer = answer_data[i]["pred"]
                answer = self.normalize_answer(answer)
                correct_answer = problem["answer"]
                correct_answer = self.normalize_answer(correct_answer)
                if correct_answer != "" and "corrected_question_index" not in problem and answer == correct_answer:
                    total_score += int(problem["points"])
                
                history_data.append({
                    "year": year,
                    "section": section,
                    "index": problem["index"],
                    "text_only": problem["text_only"],
                    "pred": answer,
                    "answer": correct_answer,
                    "points": int(problem["points"]),
                    "human_accuracy": problem["human_accuracy"]
                })
            with open(os.path.join(save_path, "保健", f"保健_{year}_{section.lower()}_history.json"), "w") as f:
                json.dump(history_data, f, indent=4)
        
        if total_score >= 87:
            pass_or_not = True
        
        return total_score, pass_or_not, False
    
    def rigaku_score(self, year, answer_res_path, save_path, fix_format=False):
        """score the result of the LLMs on 理学療法士国試"""
        pass_scores = [168, 43]

        def get_ryouiki(index):
            index = int(index.split("-")[0])
            if index <= 80: # 必修
                return 0
            else: # 実地
                return 1

        total_score = [0, 0] # [必修、一般]
        pass_or_not = False

        # read the ground truth data (correct answer, points, forbidden choices, etc.)
        ground_truth_path = os.path.join(self.data_dir, "理学")

        for section in ["A", "B"]:
            # read the ground truth data
            with open(os.path.join(ground_truth_path, f"理学_{year}_{section.lower()}.json"), "r") as f:
                question_data = json.load(f)
            
            # read the answer of the LLM
            with open(os.path.join(answer_res_path, "理学", f"理学_{year}_{section.lower()}_pred.json"), "r") as f:
                answer_data = json.load(f)
            
            history_data = []
            # score the answer of the LLM
            for i, problem in enumerate(question_data):
                answer = answer_data[i]["pred"]
                answer = self.normalize_answer(answer)
                correct_answer = problem["answer"]
                correct_answer = self.normalize_answer(correct_answer)
                score_index = get_ryouiki(problem["index"])
                if correct_answer != "" and "corrected_question_index" not in problem and answer == correct_answer:
                    total_score[score_index] += int(problem["points"])
                
                history_data.append({
                    "year": year,
                    "section": section,
                    "index": problem["index"],
                    "text_only": problem["text_only"],
                    "pred": answer,
                    "answer": correct_answer,
                    "points": int(problem["points"]),
                    "human_accuracy": problem["human_accuracy"]
                })
            with open(os.path.join(save_path, "理学", f"理学_{year}_{section.lower()}_history.json"), "w") as f:
                json.dump(history_data, f, indent=4)
        
        if total_score[0] >= pass_scores[0] and total_score[1] >= pass_scores[1]:
            pass_or_not = True
        
        return total_score, pass_or_not, False
    
    def sagyou_score(self, year, answer_res_path, save_path, fix_format=False):
        """score the result of the LLMs on 作業療法士国試"""
        pass_scores = [168, 43]

        def get_ryouiki(index):
            index = int(index.split("-")[0])
            if index <= 80: # 必修
                return 0
            else: # 実地
                return 1

        total_score = [0, 0] # [必修、一般]
        pass_or_not = False

        # read the ground truth data (correct answer, points, forbidden choices, etc.)
        ground_truth_path = os.path.join(self.data_dir, "作業")

        for section in ["A", "B"]:
            # read the ground truth data
            with open(os.path.join(ground_truth_path, f"作業_{year}_{section.lower()}.json"), "r") as f:
                question_data = json.load(f)
            
            # read the answer of the LLM
            with open(os.path.join(answer_res_path, "作業", f"作業_{year}_{section.lower()}_pred.json"), "r") as f:
                answer_data = json.load(f)
            
            history_data = []
            # score the answer of the LLM
            for i, problem in enumerate(question_data):
                answer = answer_data[i]["pred"]
                answer = self.normalize_answer(answer)
                correct_answer = problem["answer"]
                correct_answer = self.normalize_answer(correct_answer)
                score_index = get_ryouiki(problem["index"])
                if correct_answer != "" and "corrected_question_index" not in problem and answer == correct_answer:
                    total_score[score_index] += int(problem["points"])
                
                history_data.append({
                    "year": year,
                    "section": section,
                    "index": problem["index"],
                    "text_only": problem["text_only"],
                    "pred": answer,
                    "answer": correct_answer,
                    "points": int(problem["points"]),
                    "human_accuracy": problem["human_accuracy"]
                })
            with open(os.path.join(save_path, "作業", f"作業_{year}_{section.lower()}_history.json"), "w") as f:
                json.dump(history_data, f, indent=4)
        
        if total_score[0] >= pass_scores[0] and total_score[1] >= pass_scores[1]:
            pass_or_not = True
        
        return total_score, pass_or_not, False

    def jyosan_score(self, year, answer_res_path, save_path, fix_format=False):
        """score the result of the LLMs on 助産師国試"""
        total_score = 0
        pass_or_not = False

        # read the ground truth data (correct answer, points, forbidden choices, etc.)
        ground_truth_path = os.path.join(self.data_dir, "助産")

        for section in ["A", "B"]:
            # read the ground truth data
            with open(os.path.join(ground_truth_path, f"助産_{year}_{section.lower()}.json"), "r") as f:
                question_data = json.load(f)
            
            # read the answer of the LLM
            with open(os.path.join(answer_res_path, "助産", f"助産_{year}_{section.lower()}_pred.json"), "r") as f:
                answer_data = json.load(f)
            
            history_data = []
            # score the answer of the LLM
            for i, problem in enumerate(question_data):
                answer = answer_data[i]["pred"]
                answer = self.normalize_answer(answer)
                correct_answer = problem["answer"]
                correct_answer = self.normalize_answer(correct_answer)
                if correct_answer != "" and "corrected_question_index" not in problem and answer == correct_answer:
                    total_score += int(problem["points"])
                
                history_data.append({
                    "year": year,
                    "section": section,
                    "index": problem["index"],
                    "text_only": problem["text_only"],
                    "pred": answer,
                    "answer": correct_answer,
                    "points": int(problem["points"]),
                    "human_accuracy": problem["human_accuracy"]
                })
            with open(os.path.join(save_path, "助産", f"助産_{year}_{section.lower()}_history.json"), "w") as f:
                json.dump(history_data, f, indent=4)
        
        if total_score >= 87:
            pass_or_not = True
        
        return total_score, pass_or_not, False

    def shinryo_score(self, year, answer_res_path, save_path, fix_format=False):
        """score the result of the LLMs on 診療放射線技師国試"""
        pass_score = 120
        total_score = 0
        pass_or_not = False

        # read the ground truth data (correct answer, points, forbidden choices, etc.)
        ground_truth_path = os.path.join(self.data_dir, "診療")

        for section in ["A", "B"]:
            # read the ground truth data
            with open(os.path.join(ground_truth_path, f"診療_{year}_{section.lower()}.json"), "r") as f:
                question_data = json.load(f)
            
            # read the answer of the LLM
            with open(os.path.join(answer_res_path, "診療", f"診療_{year}_{section.lower()}_pred.json"), "r") as f:
                answer_data = json.load(f)
            
            history_data = []
            # score the answer of the LLM
            for i, problem in enumerate(question_data):
                answer = answer_data[i]["pred"]
                answer = self.normalize_answer(answer)
                correct_answer = problem["answer"]
                correct_answer = self.normalize_answer(correct_answer)
                if correct_answer != "" and "corrected_question_index" not in problem and answer == correct_answer:
                    total_score += int(problem["points"])
                
                history_data.append({
                    "year": year,
                    "section": section,
                    "index": problem["index"],
                    "text_only": problem["text_only"],
                    "pred": answer,
                    "answer": correct_answer,
                    "points": int(problem["points"]),
                    "human_accuracy": problem["human_accuracy"]
                })
            with open(os.path.join(save_path, "診療", f"診療_{year}_{section.lower()}_history.json"), "w") as f:
                json.dump(history_data, f, indent=4)
        
        if total_score >= pass_score:
            pass_or_not = True
        
        return total_score, pass_or_not, False
        
    def shinou_score(self, year, answer_res_path, save_path, fix_format=False):
        """score the result of the LLMs on 視能訓練士国試"""
        pass_score = 102
        total_score = 0
        pass_or_not = False

        # read the ground truth data (correct answer, points, forbidden choices, etc.)
        ground_truth_path = os.path.join(self.data_dir, "視能")

        for section in ["A", "B"]:
            # read the ground truth data
            with open(os.path.join(ground_truth_path, f"視能_{year}_{section.lower()}.json"), "r") as f:
                question_data = json.load(f)
            
            # read the answer of the LLM
            with open(os.path.join(answer_res_path, "視能", f"視能_{year}_{section.lower()}_pred.json"), "r") as f:
                answer_data = json.load(f)
            
            history_data = []
            # score the answer of the LLM
            for i, problem in enumerate(question_data):
                answer = answer_data[i]["pred"]
                answer = self.normalize_answer(answer)
                correct_answer = problem["answer"]
                correct_answer = self.normalize_answer(correct_answer)
                if correct_answer != "" and "corrected_question_index" not in problem and answer == correct_answer:
                    total_score += int(problem["points"])
                
                history_data.append({
                    "year": year,
                    "section": section,
                    "index": problem["index"],
                    "text_only": problem["text_only"],
                    "pred": answer,
                    "answer": correct_answer,
                    "points": int(problem["points"]),
                    "human_accuracy": problem["human_accuracy"]
                })
            with open(os.path.join(save_path, "視能", f"視能_{year}_{section.lower()}_history.json"), "w") as f:
                json.dump(history_data, f, indent=4)
        
        if total_score >= pass_score:
            pass_or_not = True
        
        return total_score, pass_or_not, False
    
    def yakuzai_score(self, year, answer_res_path, save_path, fix_format=False):
        """score the result of the LLMs on 薬剤師国試"""
        pass_score_dict = {
            "2020": 426, # total score
            "2021": 430, 
            "2022": 434, 
            "2023": 470, 
            "2024": 420,  
        }
        must_score_line = 126 # 必修 >70% correct
        area_score_ratio = 0.3 # each subject >30% correct in must

        total_score = 0
        must_score = 0
        area_score = {} # actual score of each subject in must
        area_total_score = {} # total score of each area
        pass_or_not = False

        # read the ground truth data (correct answer, points, forbidden choices, etc.)
        ground_truth_path = os.path.join(self.data_dir, "薬剤")

        for section in ["a1", "a2", "a3", "b1", "b2", "b3"]:
            # read the ground truth data
            with open(os.path.join(ground_truth_path, f"薬剤_{year}_{section.lower()}.json"), "r") as f:
                question_data = json.load(f)
            
            # read the answer of the LLM
            with open(os.path.join(answer_res_path, "薬剤", f"薬剤_{year}_{section.lower()}_pred.json"), "r") as f:
                answer_data = json.load(f)
            
            history_data = []
            # score the answer of the LLM
            for i, problem in enumerate(question_data):
                answer = answer_data[i]["pred"]
                answer = self.normalize_answer(answer)
                correct_answer = problem["answer"]
                correct_answer = self.normalize_answer(correct_answer)
                if correct_answer != "" and "corrected_question_index" not in problem:
                    area = problem["answer_sub2"]
                    if area not in area_score:
                        area_score[area] = 0
                        area_total_score[area] = 0

                    if answer == correct_answer:
                        total_score += int(problem["points"])
                        area_total_score[area] += int(problem["points"])
                        if section == "a1": # this is the must section
                            must_score += int(problem["points"])
                        else:
                            area_score[area] += int(problem["points"])
                
                history_data.append({
                    "year": year,
                    "section": section,
                    "index": problem["index"],
                    "text_only": problem["text_only"],
                    "subject": problem["answer_sub2"],
                    "pred": answer,
                    "answer": correct_answer,
                    "points": int(problem["points"]),
                    "human_accuracy": problem["human_accuracy"]
                })
            with open(os.path.join(save_path, "薬剤", f"薬剤_{year}_{section.lower()}_history.json"), "w") as f:
                json.dump(history_data, f, indent=4, ensure_ascii=False)
        
        if total_score >= pass_score_dict[str(year)] and must_score >= must_score_line and all(area_score[area] >= area_score_ratio * area_total_score[area] for area in area_score):
            pass_or_not = True
        
        score_record = {
            "total_score": total_score,
            "must_score": must_score,
            "area_score": area_score,
            "area_total_score": area_total_score
        }
        return score_record, pass_or_not, False
            
    
    def score(self, test_type, year, answer_res_path, save_path, fix_format=False):
        if test_type == "医師":
            return self.ishi_score(year, answer_res_path, save_path, fix_format)
        elif test_type == "歯科":
            return self.shika_score(year, answer_res_path, save_path, fix_format)
        elif test_type == "看護":
            return self.kango_score(year, answer_res_path, save_path, fix_format)
        elif test_type == "保健":
            return self.hoken_score(year, answer_res_path, save_path, fix_format)
        elif test_type == "理学":
            return self.rigaku_score(year, answer_res_path, save_path, fix_format)
        elif test_type == "作業":
            return self.sagyou_score(year, answer_res_path, save_path, fix_format)
        elif test_type == "助産":
            return self.jyosan_score(year, answer_res_path, save_path, fix_format)
        elif test_type == "診療":
            return self.shinryo_score(year, answer_res_path, save_path, fix_format)
        elif test_type == "視能":
            return self.shinou_score(year, answer_res_path, save_path, fix_format)
        elif test_type == "薬剤":
            return self.yakuzai_score(year, answer_res_path, save_path, fix_format)
        else:
            raise ValueError(f"Invalid test type: {test_type}")

    def total_scores(self, company, model, input_type, fix_format=False):
        """score the result of the LLMs on all the exams and save the result to a csv file
        Args:
            fix_format (bool): if True, fix the format of the answer of the problems
        """
        answer_res_path = os.path.join(self.res_dir, company, model, input_type)
        assert os.path.exists(answer_res_path), f"The result of {company} {model} {input_type} does not exist"
        save_path = os.path.join(self.score_dir, company, model, input_type)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for test_type in tqdm(TEST_TYPE_MAP.keys()):
            test_results = []
            print(f"Scoring {test_type}")
            if not os.path.exists(os.path.join(save_path, test_type)):
                os.makedirs(os.path.join(save_path, test_type))

            for year in range(2020, 2025):
                if test_type == "薬剤":
                    score_record, pass_or_not, failed_by_forbidden = self.score(test_type, year, answer_res_path, save_path, fix_format)
                    test_result = {
                        "test_type": test_type,
                        "year": year,
                        "total_score": score_record["total_score"],
                        "must_score": score_record["must_score"],
                    }
                    area_keys = list(score_record["area_score"].keys())
                    for area in area_keys:
                        test_result[area] = score_record["area_score"][area]
                        test_result[area + "_total"] = score_record["area_total_score"][area]
                    test_result["pass_or_not"] = pass_or_not
                    test_result["failed_by_forbidden"] = failed_by_forbidden
                    test_results.append(deepcopy(test_result))
                else:
                    total_score, pass_or_not, failed_by_forbidden = self.score(test_type, year, answer_res_path, save_path, fix_format)
                    must_score = 0 if isinstance(total_score, int) else total_score[0]
                    total_score = total_score if isinstance(total_score, int) else sum(total_score)

                    test_results.append({
                        "test_type": test_type,
                        "year": year,
                        "total_score": total_score,
                        "must_score": must_score,
                        "pass_or_not": pass_or_not,
                        "failed_by_forbidden": failed_by_forbidden
                    })
        
            df = pd.DataFrame(test_results)
            df.to_csv(os.path.join(save_path, test_type, "total_scores.csv"), index=False)


if __name__ == "__main__":
    if not os.path.exists("./scoring"):
        os.makedirs("./scoring")

    scoring = Scoring("./results", "./scoring", "./exams/JA") # TODO: add the passing scores for each test
    for company in os.listdir("./results"):
        for model in os.listdir(os.path.join("./results", company)):
            for input_type in os.listdir(os.path.join("./results", company, model)): # text or multimodal
                print(f"Scoring {company} {model} {input_type}")
                scoring.total_scores(company, model, input_type)
    

    


