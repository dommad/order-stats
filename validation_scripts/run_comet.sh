for i in `ls 0*_42_*mzXML`;
do out=$(echo $i | cut -f 1 -d '.');
/usr/local/tpp/bin/comet -Pcomet_t_val -Dhuman_t.fasta -N${out}_pos refined_$i;
/usr/local/tpp/bin/comet -Pcomet_random -Dhuman_t.fasta -N${out}_rand $i;
/usr/local/tpp/bin/comet -Pcomet_d_val -Dhuman_td.fasta -N${out}_dec $i; done
