#!/bin/sh

#TOPDIR=/nfsvol/nlsaux16/II_Group_WorkArea/MedPost-SKR/Prod/MedPost-SKR
TOPDIR=/home/gloria/Documents/UPC/HLE/MedPost-SKR
PATH=$PATH/usr/lib64/jvm/java-8-openjdk/jre/bin/
DATADIR=${TOPDIR}/data
LEXDBFILE=${DATADIR}/lexDB.serial
NGRAMFILE=${DATADIR}/ngramOne.serial
CLASSPATH=".:${TOPDIR}/lib/mps.jar"

JVMOPTIONS="-DlexFile=${LEXDBFILE} -DngramOne=${NGRAMFILE}"
export JVMOPTIONS
export CLASSPATH

java $JVMOPTIONS -cp $CLASSPATH gov.nih.nlm.nls.mps.Tagger $*
