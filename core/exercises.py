"""مكتبة التمرينات وخطة الوحدات الاثنتي عشرة.

مطابقة لمستند «مكتبة تمرينات تطوير الاستجابة الحركية». جميع القيم مقترحة وتُثبَّت
بعد عرضها على الخبراء ونتائج التجربة الاستطلاعية.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Exercise:
    code: str
    level: int
    name: str
    stimulus: str
    response: str
    sets: int = 3
    reps: int = 6
    skills: tuple = ()          # speed / choice / anticipation / accuracy / technique
    stim: dict = field(default_factory=dict)  # إعدادات مولّد المثيرات
    contact: bool = False       # هل يوجد ضرب بالمضرب (قياس الوضع الحركي)؟
    use_wrist: bool = False     # هل تبدأ الاستجابة بحركة الذراع؟


def _s(mode, dmin=1.0, dmax=3.0, **kw):
    d = {"mode": mode, "delay_min": dmin, "delay_max": dmax}
    d.update(kw)
    return d


LIBRARY: list[Exercise] = [
    Exercise("1-1", 1, "انطلاق إلى علامة واحدة", "لون واحد على الشاشة",
             "انطلاق من وضع الاستعداد إلى علامة أمام الشبكة ثم العودة للمركز",
             3, 6, ("speed",), _s("single_color")),
    Exercise("1-2", 1, "خطوة التهيؤ (Split step)", "إشارة صوتية",
             "قفزة قصيرة مع ثبات وضع الاستعداد", 3, 8, ("speed",), _s("sound")),
    Exercise("1-3", 1, "لمس علامة بالمضرب", "إشارة لونية",
             "مد الذراع ولمس علامة أمامية ثم العودة", 3, 6, ("speed", "technique"),
             _s("single_color"), use_wrist=True),
    Exercise("2-1", 2, "انطلاق لأربع زوايا حسب اللون", "4 ألوان، لكل لون زاوية",
             "انطلاق إلى الزاوية المقابلة للون ثم العودة", 3, 8, ("choice",),
             _s("four_colors")),
    Exercise("2-2", 2, "انطلاق حسب السهم", "سهم يمين/يسار/أمام/خلف",
             "انطلاق في اتجاه السهم ثم العودة", 3, 8, ("choice",), _s("arrows")),
    Exercise("2-3", 2, "انطلاق حسب الرقم", "رقم من 1 إلى 6 يمثل منطقة من الملعب",
             "الوصول إلى المنطقة المرقمة", 3, 6, ("choice",), _s("numbers")),
    Exercise("3-1", 3, "قراءة حركة المدرب", "بدء حركة المدرب قبل ضرب الريشة",
             "توقع الاتجاه والتحرك قبل خروج الريشة", 3, 6, ("anticipation",),
             _s("coach_cue", 1.5, 3.5)),
    Exercise("3-2", 3, "مثير خادع", "إشارة تتغير في آخر لحظة",
             "تعديل الاتجاه أثناء الحركة", 3, 6, ("anticipation", "choice"),
             _s("fake_arrow", fake_prob=0.4)),
    Exercise("3-3", 3, "إشارتان متتاليتان", "إشارة تهيؤ ثم إشارة الاتجاه بعد تأخير متغير",
             "تهيؤ ثم انطلاق فور ظهور إشارة الاتجاه", 3, 6, ("anticipation", "speed"),
             _s("delayed_arrow", 1.0, 2.5)),
    Exercise("4-1", 4, "ضربة الشبكة المستقيمة على إشارة", "إشارة ثم ريشة قصيرة مغذّاة من المدرب",
             "انطلاق للشبكة وأداء ضربة شبكة مستقيمة إلى منطقة الهدف", 3, 6,
             ("accuracy", "technique", "speed"), _s("skill_cue", skill="net_straight"),
             contact=True, use_wrist=True),
    Exercise("4-2", 4, "ضربة الشبكة القطرية على إشارة", "إشارة ثم ريشة قصيرة مغذّاة من المدرب",
             "انطلاق للشبكة وأداء ضربة شبكة قطرية إلى المنطقة المقابلة", 3, 6,
             ("accuracy", "technique"), _s("skill_cue", skill="net_cross"),
             contact=True, use_wrist=True),
    Exercise("4-3", 4, "الرفع من الشبكة على إشارة", "إشارة ثم ريشة قصيرة مغذّاة من المدرب",
             "انطلاق للشبكة ورفع الريشة عالياً إلى عمق الملعب", 3, 6,
             ("accuracy", "technique", "choice"), _s("skill_cue", skill="net_lift"),
             contact=True, use_wrist=True),
    Exercise("4-4", 4, "مهارة شبكة متغيرة حسب الإشارة", "إشارة تحدد المهارة (مستقيمة/قطرية/رفع) ومنطقتها",
             "أداء مهارة الشبكة المطلوبة نحو المنطقة المطلوبة", 3, 6,
             ("accuracy", "choice", "technique"), _s("skill_cue", skill="mixed"),
             contact=True, use_wrist=True),
]

_BY_CODE = {e.code: e for e in LIBRARY}

SKILL_LABELS = {
    "speed": "سرعة الاستجابة",
    "choice": "الاختيار بين مثيرات",
    "anticipation": "التوقع",
    "accuracy": "الدقة",
    "technique": "الأداء الفني",
}


def get_exercise(code: str) -> Exercise:
    return _BY_CODE[code]


def exercises_by_level(level: int) -> list[Exercise]:
    return [e for e in LIBRARY if e.level == level]


# (الأسبوع، الوحدة، المستويات، التمرينات، الغرض)
UNIT_PLAN: list[dict] = [
    dict(week=1, unit=1, levels=(1,), codes=("1-1", "1-2", "1-3"), goal="تعريف اللاعب بالمثيرات والأداء الأساسي"),
    dict(week=1, unit=2, levels=(1,), codes=("1-1", "1-2", "1-3"), goal="ترسيخ الاستجابة البسيطة بمثيرات أسرع"),
    dict(week=2, unit=3, levels=(1, 2), codes=("1-1", "2-1"), goal="الانتقال إلى الاختيار بين مثيرين"),
    dict(week=2, unit=4, levels=(2,), codes=("2-1", "2-2"), goal="استجابة اختيارية بأربعة اتجاهات"),
    dict(week=3, unit=5, levels=(2,), codes=("2-1", "2-2", "2-3"), goal="زيادة عدد المثيرات وتقليل زمن العرض"),
    dict(week=3, unit=6, levels=(2, 4), codes=("2-2", "2-3", "4-1"), goal="أول دمج مع ضربة الشبكة المستقيمة"),
    dict(week=4, unit=7, levels=(3, 4), codes=("3-1", "4-1", "4-2"), goal="إدخال التوقع مع ضربتي الشبكة المستقيمة والقطرية"),
    dict(week=4, unit=8, levels=(3, 4), codes=("3-1", "3-2", "4-2", "4-3"), goal="توقع ومثير خادع مع الضربة القطرية والرفع"),
    dict(week=5, unit=9, levels=(3, 4), codes=("3-2", "3-3", "4-1", "4-3"), goal="استجابة مع تأخير متغير ومهارتا المستقيمة والرفع"),
    dict(week=5, unit=10, levels=(3, 4), codes=("3-3", "4-1", "4-2", "4-3"), goal="تثبيت دمج مهارات الشبكة الثلاث"),
    dict(week=6, unit=11, levels=(3, 4), codes=("3-2", "3-3", "4-4"), goal="مهارة شبكة متغيرة عند أقصى صعوبة"),
    dict(week=6, unit=12, levels=(3, 4), codes=("3-1", "4-4"), goal="تثبيت ومراجعة قبل الاختبار البعدي"),
]

N_UNITS = len(UNIT_PLAN)


def plan_for_unit(unit: int) -> dict:
    for p in UNIT_PLAN:
        if p["unit"] == unit:
            return p
    raise KeyError(unit)


def exercises_for_unit(unit: int) -> list[Exercise]:
    return [get_exercise(c) for c in plan_for_unit(unit)["codes"]]


# ------------------------------------------------------------ مستوى الصعوبة 1..5
DIFFICULTY_MIN, DIFFICULTY_MAX, DIFFICULTY_DEFAULT = 1, 5, 3


def apply_difficulty(ex: Exercise, difficulty: int) -> dict:
    """يعيد معاملات التنفيذ بعد ضبط الصعوبة (تُستخدم في مجموعة الشبكات العصبية العميقة)."""
    d = max(DIFFICULTY_MIN, min(DIFFICULTY_MAX, int(difficulty)))
    shift = d - DIFFICULTY_DEFAULT
    stim = dict(ex.stim)
    stim["delay_max"] = round(max(stim["delay_min"] + 0.5, stim["delay_max"] - 0.25 * shift), 2)
    if "fake_prob" in stim:
        stim["fake_prob"] = round(min(0.7, max(0.2, stim["fake_prob"] + 0.1 * shift)), 2)
    reps = int(max(4, min(10, ex.reps + shift)))
    rest_rep = 20 - 2 * shift            # ثانية بين التكرارات
    rest_set = 90 - 10 * shift           # ثانية بين المجموعات
    return dict(code=ex.code, name=ex.name, sets=ex.sets, reps=reps,
                rest_rep=rest_rep, rest_set=rest_set, stim=stim, difficulty=d)


def default_prescription(unit: int) -> list[dict]:
    """البرنامج الثابت (مجموعة الرؤية الحاسوبية): نفس الجدول لكل اللاعبين."""
    return [apply_difficulty(e, DIFFICULTY_DEFAULT) for e in exercises_for_unit(unit)]


def allowed_moves(ex: "Exercise") -> tuple:
    """أنواع الحركة الممكنة للتمرين: ضربات الشبكة (المستوى 4) تحتاج مضرباً على الأقل."""
    return ("racket", "racket_shuttle") if ex.level == 4 else ("free", "racket", "racket_shuttle")
