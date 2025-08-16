PASJ_Proc_Template
加速器学会年会 Proceedings template LaTeX 版
2023.6.29.

Tex Live を用いて、PDF を生成するためには次の Command が使用できます。

TeX Live
https://www.tug.org/texlive/
https://www.tug.org/mactex/

LuaTeX による PDF 生成 
latexmk -lualatex TeXtemplate2023.tex

upTeX による PDF 生成 
latexmk -pdfdvi TeXtemplate2023.tex
(upTeX の処理には同じ Directory にある latexmkrc が必要です)


(1) 2020, 2022 年版からの変更点

a) LuaTeX 版と upTeX 版の Template を合体させた
  (iftex package を用いた upLaTeX 版と LuaLaTeX 版の統合)
b) Word 版 Template の文言と整合させた
c) subsubsection header を Word 版に似せた
d) 著者 Email と所属を指定する場合の書式の調整
e) Tex Live, Cloud LaTeX, Overleaf の説明の調整
など

(2) Cloud service の利用例

Cloud LaTeX と Overleaf の Template として登録する予定です。
使用方法は次のようになると思われます。

[ Overleaf ]
　Template page から PASJ_Proc_Template_2023 を検索して使用する

　Menu から Compiler として LaTeX (upLaTeX) または LuaLaTeX が選べます

[ Cloud LaTeX ]
　My Page において
　　テンプレートから作成 (LOAD TEMPLATE) を選択
　　　PASJ_Proc_Template_2023 を選択

　Menu から LaTeX エンジンとして uplatex または lualatex が選べます

(4) Source の Download

当面、上の Template と同じものを次の場所にも置いてあります

http://www-linac.kek.jp/temp/pasj/PASJ_Proc_Template_2023.zip

===
古川 和朗 <kazuro.furukawa@kek.jp>
