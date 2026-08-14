# Data and document licensing

The code in this repository is Apache-2.0 (`LICENSE`). The documents it was
run over are not, and several different terms apply. This file records them.

---

## 1. UK tribunal judgments — not published here

The Employment Appeal Tribunal judgments used in Sections 5.3 and 5.6 are
**Crown copyright**, published by The National Archives through the Find Case
Law service.

The service's standard Open Justice Licence permits quotation and citation and
**excludes computational analysis**, which The National Archives defines to
include processing judgments with large language models. The research was
therefore carried out under a **transactional re-use licence granted to Leiden
University by The National Archives (June 2026)** for this project.

Two consequences:

1. **That licence covers the research. It does not transfer to you.** If you
   want to re-run the experiments that read judgments, apply to The National
   Archives for your own licence.
2. **No judgment text appears in this repository**, in any form: not as
   corpora, not inside prompts, not as quoted sentences in result files. Free
   text cannot be reliably de-identified, and it is licensed material
   regardless of that.

Results are keyed by neutral citation. A neutral citation contains no personal
name and retrieves the exact judgment on Find Case Law, so every computed date
stays checkable against the identified public source.

> Crown copyright material is reproduced by permission of The National
> Archives. The contents of the judgments can be used under the Open Justice
> Licence.
>
> The licensed material only partially represents the activities of the courts
> and tribunals. The cases studied reached the Employment Appeal Tribunal
> because their timeliness was contested, and they are a selected sample of
> tribunal activity.

**Personal data.** The licence is neither a data-sharing agreement nor a
processing agreement for personal data in the licensed material. Party names
are personal data. No party names are published here. One case was
additionally anonymised by the tribunal itself, and that anonymisation is
preserved.

---

## 2. Perturbed judgments — generator published, output not

The counterfactual harness in Section 5.6 produces variants of real judgments
which, read alone, make false date claims about real people's cases. Published
here: the generator and the per-item shift offsets. Not published: the
generated set. A licensed user can re-derive every item from the published
judgments.

---

## 3. Statutes

| Source | Terms |
|---|---|
| UK provisions from legislation.gov.uk | Open Government Licence v3.0 |
| Dutch statutes | no copyright (Auteurswet, art. 11) |
| Court of Justice of the European Union material | Commission re-use decision 2011/833/EU |

---

## 4. Contracts

The extraction and verification experiments use contracts sampled from
Multi\_Legal\_Pile, under that dataset's own terms. Sample seeds are recorded
so the selection can be reproduced; the contract texts themselves are not
redistributed here.

---

## 5. Third-party benchmarks

DeonticBench and TRACIE are used in Section 5.5 and are **not vendored** in
this repository. Fetch each from its own source under its own licence. TRACIE
is Apache-2.0.

---

## 6. Fabricated example documents

All example documents shipped with the tool, and all documents in its test
suite, are invented. No real case, party or filing is described or implied.
They exist so the tool can be run end to end without any licensed material.

---

## If you are unsure

The safe default is that any document this repository was run over is licensed
material and any text extracted from one is too. Publish neither. Publish the
computed metadata, keyed by citation, and let a reader retrieve the source
themselves.
