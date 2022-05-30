for i in `ls refined*5*.mzXML`;

do /data/dominik/comet/comet.2021010.linux.exe -Pcomet_21_pos $i -N${i}_pos;done
#java -Xmx50g -jar /data/dominik/lower_order/validation/MSFragger-3.4/MSFragger-3.4.jar fragger.params $i;

#do core=$(echo $i | cut -d '.' -f 1);

#/tools/tpp/bin/xinteract -dDECOY -p0 -OAPd -PPM -N${core}_comet ${core}.pep.xml;
#/tools/tpp/bin/xinteract -dDECOY -p0 -OAPd -PPM -N${core}_msfrag ${core}.pepXML;

#python3 refine.py interact-${core}_comet.pep.xml interact-${core}_msfrag.pep.xml $core; 
#python3 shift_pmz.py ${core}.pkl $i; done

