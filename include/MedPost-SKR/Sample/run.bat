set path=%path%;C:\Program Files (x86)\Java\jre6\bin
set TOPDIR=D:\Jim\Z\MedPost_SKR\MedPost-SKR

set DATADIR=%TOPDIR%\data
set LEXDBFILE=%DATADIR%\lexDB.serial
set NGRAMFILE=%DATADIR%\ngramOne.serial
set CLASSPATH=".;%TOPDIR%\lib\mps.jar"

set JVMOPTIONS=-DlexFile=%LEXDBFILE% -DngramOne=%NGRAMFILE%

java %JVMOPTIONS% -cp %CLASSPATH% Test %1 %2
