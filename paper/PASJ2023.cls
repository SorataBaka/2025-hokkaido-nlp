%%
%%Class file for PASJ2020
%% merged two version for upTeX and LuaTeX
%%
%
%
%
%% Class file for Linear Accelerator Meeting in Japan, 2004
%% ad-hoc patch by R. Hajima June 21, 2004.
%% Modified for 2008 PASJ by A. Mizuno/SPring-8, Aug 01, 2008 
%% Modified for PASJ2013 by K. Yanagida/SPring-8, April 08, 2013
%% Modified for PASJ2016 by K. Yanagida/SPring-8, July 14, 2016
%% Modified for PASJ2018_LuaTeX/upTeX by K. Furukawa/KEK, Aug. 17, 2017
%% Modified for PASJ2019_LuaTeX/upTeX by K. Yanagida/SPring-8, Aug. 22, 2018
%% Modified for PASJ2019_LuaTeX/upTeX by K. Yanagida/SPring-8, Aug. 29, 2018
%% Modified for PASJ2019_LuaTeX/upTeX by K. Furukawa/KEK, May 21, 2020
%% Merged PASJ2020_LuaTeX/upTeX by K. Furukawa/KEK, Jul. 24, 2020
%% Merged/Modified PASJ2022_LuaTeX/upTeX by K. Furukawa/KEK, Apr. 24, Jun. 26, 2023
%
%
%
%% This file has been developed as a common template for papers
%% destined for electronic production using Adobe Acrobat
%% software for Accelerator Conferences
%%
%% See the JACo Website for more information
%% 
%%       http://jacow.org/
%%
%
%
%
%%  Special thanks to John Jowett and Michel Goossens from CERN and
%%  Martin Comyn at TRIUMF for their significant contributions to 
%%  this class file over the period 1996 to 2000.
%%                                                 John Poole
%%                                                 March 2000
%
%
%
%%  JAC2001.cls is a modified version of JAC2000.cls to produce indented
%%  first paragraphs after section, subsection and subsubsection headings.
%%
%%                                                 Martin Comyn  April 2001

\def\fileversion{1.4}
\def\filedate{2023/06/26}
\def\docdate {2023/06/26}

\newlength{\dsep}
\newlength{\dwid}
\newlength{\dht}
\NeedsTeXFormat{LaTeX2e}
%\ProvidesClass{JAC2003}[\filedate\space Version \fileversion]
%\ProvidesClass{PASJ2016}[\filedate\space Version \fileversion]
%\ProvidesClass{PASJ2019_LuaTeX/upTeX}[\filedate\space Version \fileversion]
\ProvidesClass{PASJ2023}[\filedate\space Version \fileversion]
\DeclareOption{acus}
   {\setlength{\dwid}{6mm}%
    \setlength{\dht}{-18mm}
    \setlength{\paperheight}{11in}
    \setlength{\paperwidth}{8.5in}
    \typeout{Setup for US LETTER PAPER}}
\DeclareOption{aca4}
   {\setlength{\dwid}{0mm}%
    \setlength{\dht}{0mm}}
\DeclareOption{boxit}
%%
%%  This option draws a box on the output text with the correct margins
%%
   {\special{!userdict begin /bop-hook{
         gsave
         newpath
         58 55 moveto
         58 737 lineto
         539 737 lineto
         539 55 lineto
         closepath
         stroke
         grestore
         }def end}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ExecuteOptions{aca4}
\ProcessOptions

%The following fontsizes and linespaces are modified for PASJ.

% versions for LuaTeX and upTeX were merged, K.F
\usepackage{iftex}
%\LoadClass[10pt,a4paper]{article}
\ifLuaTeX  % LuaTeX
\LoadClass[10pt,a4paper]{ltjsarticle}
\RequirePackage{graphicx}
% 5% smaller than those of upTeX, a compromise between several possibilities, K.F
% a balance between japanese fonts and latin fonts is wrong
\renewcommand{\normalsize}{\fontsize{9.54pt}{14pt}\selectfont}
\renewcommand{\large}{\fontsize{11.45pt}{16.7pt}\selectfont}
\renewcommand{\Large}{\fontsize{13.36pt}{19.6pt}\selectfont}
%\renewcommand{\small}{\fontsize{8.55pt}{12.6pt}\selectfont}
%Define line spaces for an author list and a table.
\newcommand{\authorf}{\fontsize{9.54pt}{16pt}\selectfont}
\newcommand{\tabsmall}{\fontsize{8.58pt}{16pt}\selectfont}
\newcommand{\citesize}{\fontsize{7.63pt}{14pt}\selectfont}
\else  % upTeX
% \LoadClass[10pt,a4paper,uplatex,dvipdfmx]{jsarticle}
\LoadClass[10pt,a4paper,uplatex]{jsarticle}
% \RequirePackage{graphicx}
\RequirePackage[dvipdfmx]{graphicx}
\renewcommand{\normalsize}{\fontsize{10.04pt}{14pt}\selectfont}
\renewcommand{\large}{\fontsize{12.05pt}{16.7pt}\selectfont}
\renewcommand{\Large}{\fontsize{14.06pt}{19.6pt}\selectfont}
%\renewcommand{\small}{\fontsize{9pt}{12.6pt}\selectfont}
%Define line spaces for an author list and a table.
\newcommand{\authorf}{\fontsize{10.04pt}{16pt}\selectfont}
\newcommand{\tabsmall}{\fontsize{9.03pt}{16pt}\selectfont}
\newcommand{\citesize}{\fontsize{8.03pt}{14pt}\selectfont}
\fi  % LuaTeX/upTeX

\renewcommand{\figurename}{Figure~}
\renewcommand{\tablename}{Table~}

%%%%% Yanagida added following five lines. (2018/08/29) %%%%%
\RequirePackage{subfig}
\RequirePackage{booktabs}
\RequirePackage{url}
\RequirePackage{cite}
\RequirePackage{secdot}
\RequirePackage{textcase}
%%%%% Yanagida added above five lines. (2018/08/29) %%%%%
%%
%%  Using Times fonts produces a Uniform, compact document
%%  suitable for the Acrobat Software and printable anywhere.
%%
%%  In a 'modern' TeX installation 
%%  e.g TeXLive Version 3 or later (more information on the JACoW Website)
%%  one can use the following command
%%

\renewcommand{\rmdefault}{ptm}
\renewcommand{\sfdefault}{phv}

%%
%%  Older installations may require
%%              \RequirePackage{times}
%%
\typeout{------------------------------------------------------------------------}
\typeout{LaTeX2e Class file for Accelerator Conference publication for LaTeX2e users}
\typeout{Use the boxit option to draw a box on page showing the correct margins}
\typeout{Itemize, Enumerate and Description environments are compact versions}
\typeout{ }
\typeout{Do not use this Class file for a paper written in English.}
\typeout{------------------------------------------------------------------------}
\typeout{ }
 
%
% DIMENSION OF TEXT:
 
\setlength{\textheight}{239mm}
\setlength{\textwidth}{169.5mm}
\setlength{\topmargin}{23.75mm}
\addtolength{\topmargin}{\dht}
\setlength{\oddsidemargin}{20.25mm}
\setlength{\evensidemargin}{20.25mm}
\setlength{\columnsep}{5mm}

 
\hoffset -25.4mm
\voffset -24.4mm
 
\columnseprule 0pt      %    Width of rule between columns.
 
% Use Michel Goossens' dense lists

\RequirePackage{enumitem}
\newenvironment{Enumerate}{\begin{enumerate}[nosep]}{\end{enumerate}}
\newenvironment{Itemize}{\begin{itemize}[nosep]}{\end{itemize}}
\newenvironment{Description}{\begin{description}[nosep]}{\end{description}}

%Floatparameter:
\renewcommand{\topfraction}{.95}
\renewcommand{\bottomfraction}{.95}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.95}
 
\renewcommand{\baselinestretch}{0.85}

\def\@oddhead{}\def\@evenhead{}
\def\@oddfoot{}
\def\@evenfoot{\@oddfoot}
 
\renewcommand\thefootnote{\arabic{footnote}}
 
%-----------------------------------------------------------------------
% We acknowledge the borrowing of the clever part of
% Steven Gildea's definition of \section
% from geophysi.sty.
%
% \section is tricky because we want to both support \section* and have
% the section title uppercased.

\def\acknow{\renewcommand{\thesection}{\textsf{\@arabic\c@section}}\secdef\@aknsectionb\@aknsections}
\def\@aknsectionb[#1]#2{\@aknsections{#2}}
\def\@aknsections#1{\@startsection{section}{1}{\z@}{2.0ex plus 0.6ex minus .1ex}{1.2ex plus .2ex}{\large\gtfamily\sffamily}*{\uppercase{#1}}}

\def\section{\renewcommand{\thesection}{\textsf{\@arabic\c@section ~}}\secdef\@sectionb\@sections}
\def\@sectionb[#1]#2{\@sections{#2}}
\def\@sections#1{\@startsection{section}{1}{\z@}{2.0ex plus 0.6ex minus .1ex}{1.2ex plus .2ex}{\large\gtfamily\sffamily}{\uppercase{#1}}}

%                {\large\gtfamily}{\uppercase{#1}}}
%                {\large\bf\boldmath\centering}{\uppercase{#1}}}
% Should there be any problem with this, we can revert to the following
% but then the onus is on the user to type the argument of \section in
% uppercase in the input file.
%
%\def\section{\@startsection {section}{1}{\z@}{2.8ex plus 0.8ex minus.1ex}
%                          {1.5ex plus .2ex}{\large\bf\boldmath\centering}}
 
\def\subsection{\renewcommand{\thesection}{\@arabic\c@section}%
\@startsection{subsection}{2}{\z@}{1.6ex plus .4ex minus
   .17ex}{1.0ex plus .17ex}{\normalsize}}
% \def\subsubsection{\renewcommand{\thesection}{\@arabic\c@section}%
% \@startsection{subsubsection}{3}{\parindent}
%  {2.0ex plus .4ex minus .17ex}{-1em}{\normalsize\bfseries}}
\def\subsubsection{\renewcommand{\thesection}{\@arabic\c@section}%
\@startsection{subsubsection}{3}{\z@}
 {1.0ex plus .4ex minus .17ex}{0.5ex plus .1ex}{\normalsize}}
\def\paragraph{\@startsection
     {paragraph}{4}{\z@}{2.0ex plus .4ex minus .17ex}{-1em}{\normalsize\itshape}}
\def\subparagraph{\@startsection
     {subparagraph}{4}{\parindent}{2.0ex plus .4ex minus
     .17ex}{-1em}{\normalsize\bfseries}}
 
%\setcounter{secnumdepth}{0}
 
% This definition of \maketitle taken from article.sty, and has been
% somewhat modified.
% Small modification made by H. Hanaki for PASJ

\def\jtitle#1{\gdef\@jtitle{#1}}
\def\@jtitle{\@latex@error{No \noexpand\jtitle given}\@ehc}

\def\jauthor#1{\gdef\@jauthor{#1}}
\def\@jauthor{\@latex@error{No \noexpand\jtitle given}\@ehc}

\def\Abstract#1{\gdef\@Abstract{#1}}
\def\@Abstract{\@latex@warning@no@line{No \noexpand\Abstract given}}
 
\def\maketitle
{
  \par
  \begingroup
  \def\thefootnote{\fnsymbol{footnote}}
  \def\@makefnmark{\hbox to 5pt{$^{\@thefnmark}$\hss}}
  \twocolumn[\@maketitle]
  \@thanks
  \endgroup
  \setcounter{footnote}{0}
  \let\maketitle\relax
  \let\@maketitle\relax
  \gdef\@thanks{}\gdef\@jtitle{}\gdef\@Abstract{}\gdef\@jauthor{}\gdef\@author{}\gdef\@title{}\let\thanks\relax
}
 
\newlength{\titleblockheight}       % so user can change it if need be
\setlength{\titleblockheight}{3.5cm}
 
\newlength\titleblockstartskip
\setlength\titleblockstartskip{3pt}
\newlength\titleblockmiddleskip
\setlength\titleblockmiddleskip{1em}
\newlength\titleblockendskip
\setlength\titleblockendskip{1em}

\def\@maketitle
{\vskip \titleblockstartskip 
 {\centering
  {\Large{\gtfamily{\sffamily{\@jtitle \par}}}}
  \vskip \titleblockstartskip
  {\large\bfseries \MakeTextUppercase{\@title} \par}
  \vskip \titleblockmiddleskip               % Vertical space after title.
  {\authorf\begin{tabular}[t]{@{}c@{}}\@jauthor \end{tabular}\par}
  {\authorf\begin{tabular}[t]{@{}c@{}}\@author \end{tabular}\par}
 } %centering
 \vskip \titleblockendskip
 {\large{\textit{Abstract}}}\par ~~~~{\normalsize\@Abstract \par}
 \vfil
 % \vskip \titleblockstartskip
 \vskip \titleblockendskip
}

% The \copyrightspace command is used to produce a blank space in the first
% column where a copyright notice may go.  It works by producing a
% blank footnote of the appropriate size.  Note that it should appear
% after any \footnote commands that produce footnotes for the first
% column.   Adjusted to leave just 1cm.
 
\def\copyrightspace{\footnotetext[0]{\mbox{}\vrule height 1cm width 0pt}}

%\def\@cite#1#2{{$^{\mbox{\citesize[{#1\if@tempswa , #2\fi}]}}$}}
 
% Redefine to use smaller fonts
\def\thebibliography#1{\setlength{\itemsep}{0pt}\setlength{\parsep}{0pt}
%  \section*{REFERENCES\@mkboth
%  {REFERENCES}{REFERENCES}}\small\list
%  {\vskip 2em \large \boldmath 参考文献}\small\list
  {\vskip 1.5em \noindent \large \gtfamily {参考文献} \par}\small\list
  {[\arabic{enumi}]}
  {\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \usecounter{enumi}}
    \def\newblock{\hskip .11em plus .33em minus .07em}
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=1000\relax}
 
\let\endthebibliography=\endlist
 
\twocolumn
\sloppy
\parindent 1em
\leftmargini 2em
\leftmargin\leftmargini
\leftmarginv .5em
\leftmarginvi .5em
\flushbottom

% Use PDF V1.4 in order to embed header/footer with a certain script
% these are not necessary any more from 2023
% \ifLuaTeX
%  \directlua{pdf.setminorversion(4)}
% \else
%  % dvipdfmx -V 1.4
% \fi

%-----------------------------------------------------------------------
