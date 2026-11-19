"""Anonymise raw Dentolize exports before any analysis.

Raw exports contain personal data and are NEVER committed to this repo
(see .gitignore). Only the output of this script is published.

Usage:
    python src/anonymize.py --appointments raw/appointments/*.xlsx \
                            --payments raw/payments/*.xlsx \
                            --out data/ortho_clinic_anonymized.xlsx
"""
import argparse
import re

import numpy as np
import pandas as pd

FILE_NO = 'رقم ملف المريض'

STATUS = {'مكتمل': 'Completed', 'لم يتم الحضور': 'No-show', 'ملغي': 'Cancelled',
          'محجوز': 'Booked', 'مؤكد': 'Confirmed', 'متأخر': 'Late'}
TYPE = {'متابعة': 'Follow-up (adjustment)', 'علاج روتيني': 'Routine treatment', 'كشف': 'Examination',
        'تركيب': 'Braces placement', 'مثبت أسنان': 'Retainer', 'Separators Placement': 'Separators placement',
        'آخر': 'Other', 'إستشارة': 'Consultation', 'طارئ': 'Emergency', 'موعد تنظيف': 'Cleaning'}
METHOD = {'بطاقة بنكية': 'Card', 'نقدي': 'Cash', 'حوالة بنكية': 'Bank transfer', 'تأمين': 'Insurance', 'أخرى': 'Other'}

# Free-text cancellation reasons can contain health or personal details,
# so they are mapped to broad categories and the raw text is dropped.
CANCEL_RULES = [
    ('Travel / out of town', r'مسافر|خارج|برا بيشة|بيشة|بيشه|الرياض|جدة|أبها|مكة|الشرقية|نقل|انتقل|المنطقة|انتداب|رحلت|المملكة|السفر'),
    ('Duplicate / error / already attended', r'مكرر|حضر|خطأ|خاطئ|غلط|بالخطاء|بالغلط|الفعلي|لم يدخل|موعد اخر|موعد آخر|موعد ثاني|ماتحتاج|انتهت'),
    ('Financial', r'فلوس|مبلغ|حق|كشفيه|كشفية|دفع|رسوم|الراتب|الرواتب|المكافأة|الضمان|المادية|استرداد'),
    ('Clinic-side', r'الطبيب|براكت|البراكيت|بوند|BAND|STRIPS|المواد|وقت متاح'),
    ('No response', r'لا يوجد رد|لايوجد رد|الرقم غلط'),
    ('Health / personal circumstance', r'مستشفى|تعبان|عملية|إصابة|الولادة|جراحة|السكر|العصب|اشعة|عزا|متنوم|ظرف|ظروف'),
]


def cancel_category(text):
    if pd.isna(text):
        return None
    s = str(text).strip()
    if len(s) <= 2:
        return 'Unspecified'
    for label, pattern in CANCEL_RULES:
        if re.search(pattern, s):
            return label
    return 'Patient request / reschedule'


def to_dt(s):
    return pd.to_datetime(s, errors='coerce')


def main(appt_files, pay_files, out):
    raw = pd.concat([pd.read_excel(f, sheet_name='data') for f in appt_files], ignore_index=True)
    pay = pd.concat([pd.read_excel(f, sheet_name='data') for f in pay_files], ignore_index=True)
    raw['start'] = to_dt(raw['ابدأ'])

    # Pseudonymous IDs, ordered by first appointment; payment-only patients get Q-codes.
    order = raw.groupby(FILE_NO)['start'].min().sort_values().index
    ids = {k: f'P{i + 1:04d}' for i, k in enumerate(order)}
    for i, k in enumerate(k for k in pay[FILE_NO].unique() if k not in ids):
        ids[k] = f'Q{i + 1:04d}'

    age = (raw['start'] - to_dt(raw['تاريخ الميلاد'])).dt.days / 365.25
    appts = pd.DataFrame({
        'patient_id': raw[FILE_NO].map(ids),
        'start': raw['start'], 'end': to_dt(raw['إنهاء']), 'room': raw['غرفة'],
        'appointment_type': raw['النوع'].map(TYPE), 'status': raw['الحالة'].map(STATUS),
        'checked_in_at': to_dt(raw['تم تسجيل الوصول']), 'seen_at': to_dt(raw['تم الدخول']),
        'wait_seconds': pd.to_numeric(raw['وقت الانتظار'], errors='coerce'),
        'chair_seconds': pd.to_numeric(raw['المدة'], errors='coerce'),
        'booked_at': to_dt(raw['تم الإنشاء']),
        'cancellation_category': raw['سبب الإلغاء'].map(cancel_category),
        'age_band': pd.cut(age, [0, 13, 18, 25, 35, 200], right=False,
                           labels=['<13', '13-17', '18-24', '25-34', '35+']).astype(str).replace('nan', None),
        'nationality_group': np.where(raw['الجنسية'].isna(), None,
                                      np.where(raw['الجنسية'] == 'Saudi Arabia', 'Saudi', 'Non-Saudi')),
    }).sort_values('start').reset_index(drop=True)
    appts['lead_time_days'] = (appts.start - appts.booked_at).dt.total_seconds() / 86400
    appts.insert(0, 'appointment_id', range(1, len(appts) + 1))

    # Patient tags: keep only treatment-status tags. Charity and behaviour tags are dropped.
    g = raw.groupby(FILE_NO)
    tags = g['علامات المرضى'].first().fillna('')
    patients = pd.DataFrame({
        'patient_id': tags.index.map(ids),
        'patient_registered': to_dt(g['تاريخ إنشاء المريض'].first()).values,
        'tag_treatment_finished': tags.str.contains('FINISHED').values,
        'tag_dropped_out': tags.str.contains('منقطع').values,
        'tag_retainer': tags.str.contains('RETAINER').values,
        'tag_transfer_case': tags.str.contains('transfercase', case=False).values,
        'payment_plan': np.select([tags.str.contains('أقساط').values, tags.str.contains('مكتمل الدفع').values],
                                  ['Installments', 'Paid in full'], 'Unknown'),
        'total_paid_sar': pd.to_numeric(g['إجمالي المدفوعات'].first(), errors='coerce').values,
        'balance_sar': pd.to_numeric(g['المتبقي'].first(), errors='coerce').values,
    }).sort_values('patient_id')

    payments = pd.DataFrame({
        'patient_id': pay[FILE_NO].map(ids), 'paid_at': to_dt(pay['تم الإنشاء']),
        'amount_sar': pd.to_numeric(pay['المبلغ']), 'vat_sar': pd.to_numeric(pay['الضريبة']),
        'method': pay['النوع'].map(METHOD), 'is_bnpl_tamara': pay['آخر'].eq('تمارا'),
        'is_diagnostic_fee': pay['تفاصيل'].eq('Diagnostic Fees'), 'refunded': pay['مسترد'].eq('صحيح'),
        'patient_first_payment': to_dt(pay['تاريخ أول دفعة']),
        'patient_registered': to_dt(pay['تاريخ إنشاء المريض']),
    }).sort_values('paid_at')

    # Safety check: no phone numbers or real file numbers in the output.
    dump = pd.concat([appts.astype(str), patients.astype(str), payments.astype(str)]).to_string()
    assert not re.search(r'\+966|#\d', dump), 'Personal identifiers found in output'

    with pd.ExcelWriter(out) as w:
        appts.to_excel(w, sheet_name='appointments', index=False)
        patients.to_excel(w, sheet_name='patients', index=False)
        payments.to_excel(w, sheet_name='payments', index=False)
    print(f'Wrote {out}: {len(appts)} appointments, {len(patients)} patients, {len(payments)} payments')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--appointments', nargs='+', required=True)
    ap.add_argument('--payments', nargs='+', required=True)
    ap.add_argument('--out', default='data/ortho_clinic_anonymized.xlsx')
    a = ap.parse_args()
    main(a.appointments, a.payments, a.out)
