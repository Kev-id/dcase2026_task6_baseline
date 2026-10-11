"""Render the phase-two Markdown plan to HTML and PDF with local Chromium."""
from pathlib import Path
import subprocess
import tempfile

import markdown


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs' / '第二阶段研究计划.md'
HTML = SOURCE.with_suffix('.html')
PDF = SOURCE.with_suffix('.pdf')
STYLE = '''
@page { size: A4; margin: 17mm 15mm; }
body { font-family: "Microsoft YaHei", "SimSun", sans-serif;
       font-size: 10pt; line-height: 1.65; color: #202b38; }
h1 { font-size: 22pt; color: #173c60; border-bottom: 2pt solid #48799f; }
h2 { font-size: 14pt; color: #173c60; margin-top: 19pt; }
h3 { font-size: 11.5pt; color: #254d70; margin-top: 13pt; }
h1, h2, h3 { break-after: avoid; }
p { orphans: 3; widows: 3; }
table { width: 100%; border-collapse: collapse; font-size: 8.5pt; line-height: 1.5; }
thead { display: table-header-group; }
th { background: #e8eff5; color: #173c60; }
td, th { border: .5pt solid #bcc9d4; padding: 5pt; vertical-align: top;
         text-align: left; overflow-wrap: anywhere; }
tr { break-inside: avoid; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f2f6fa;
      padding: 8pt; font-size: 8pt; break-inside: avoid; }
code { font-family: Consolas, "Microsoft YaHei", monospace; overflow-wrap: anywhere; }
'''


def main():
    content = markdown.markdown(SOURCE.read_text(encoding='utf-8'),
                                extensions=['tables', 'fenced_code', 'sane_lists'])
    HTML.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                    '<title>第二阶段研究计划</title><style>' + STYLE +
                    '</style></head><body>' + content + '</body></html>', encoding='utf-8')
    candidates = [Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'),
                  Path('C:/Program Files/Google/Chrome/Application/chrome.exe')]
    browser = next((p for p in candidates if p.exists()), None)
    if browser is None:
        raise RuntimeError('Chromium browser unavailable; print the generated HTML to PDF.')
    with tempfile.TemporaryDirectory(prefix='phase2-pdf-') as profile:
        subprocess.run([str(browser), '--headless', '--disable-gpu', '--no-pdf-header-footer',
                        '--allow-file-access-from-files', f'--user-data-dir={profile}',
                        f'--print-to-pdf={PDF}', HTML.as_uri()], check=True, timeout=90,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert PDF.exists() and PDF.stat().st_size > 10000
    print(PDF)


if __name__ == '__main__':
    main()
