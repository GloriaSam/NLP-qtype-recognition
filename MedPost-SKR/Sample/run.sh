#!/bin/sh

TOPDIR=/nfsvol/nlsaux16/II_Group_WorkArea/MedPost-SKR/Prod/MedPost-SKR
DATADIR=${TOPDIR}/data
LEXDBFILE=${DATADIR}/lexDB.serial
NGRAMFILE=${DATADIR}/ngramOne.serial
CLASSPATH=".:${TOPDIR}/lib/mps.jar"

JVMOPTIONS="-DlexFile=${LEXDBFILE} -DngramOne=${NGRAMFILE}"
export JVMOPTIONS
export CLASSPATH

java $JVMOPTIONS -cp $CLASSPATH Test $*
