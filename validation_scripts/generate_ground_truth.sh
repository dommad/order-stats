for i in `ls 5*.mzXML`;

#do /data/dominik/comet/comet.2018014.linux.exe -Pcomet_18_old $i;
do /data/dominik/comet/comet.2021010.linux.exe -Pcomet_21_td $i;
java -Xmx50g -jar /data/dominik/lower_order/validation/MSFragger-3.4/MSFragger-3.4.jar fragger.params $i;

core=$(echo $i | cut -d '.' -f 1);

perl -i -pe 's/<search_score name="deltacn"/<search_score name="deltacnstar" value="0.0"\/>\n <search_score name="deltacn"/g;' ${core}.pep.xml;
/tools/tpp/bin/xinteract -dDECOY -p0 -OAPd -PPM -N${core}_comet ${core}.pep.xml;
/tools/tpp/bin/xinteract -dDECOY -p0 -OEAPd -PPM -N${core}_msfrag ${core}.pepXML;

python3 refine.py interact-${core}_comet.pep.xml interact-${core}_msfrag.pep.xml $core; 
python3 get_pos.py ${core}.pkl $i; done
#
