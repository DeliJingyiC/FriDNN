import os

# fricative_list contains the phonetic codes for all 8 fricatives
fricative_list = ['s', 'sh', 'f', 'th', 'z', 'zh', 'v', 'dh']
# fricative_list contains the phonetic codes for only unvoiced fricatives
unvoiced_fricative_list = ['s', 'sh', 'f', 'th']
# fricative_list contains the phonetic codes for only voiced fricatives
voiced_fricative_list = ['z', 'zh', 'v', 'dh']
# phonemes considered as silence
silence_list = ['h#', 'epi', 'pau', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl']

core_test_set_speakers = [
    'MDAB0', 'MWBT0', 'FELC0', 'MTAS1', 'MWEW0', 'FPAS0', 'MJMP0', 'MLNT0',
    'FPKT0', 'MLLL0', 'MTLS0', 'FJLM0', 'MBPM0', 'MKLT0', 'FNLP0', 'MCMJ0',
    'MJDH0', 'FMGD0', 'MGRT0', 'MNJM0', 'FDHC0', 'MJLN0', 'MPAM0', 'FMLD0'
]

val_audio_list = [
    os.path.join('DR2', 'MPDF0', 'SI2172.WAV'),
    os.path.join('DR1', 'FAKS0', 'SX133.WAV'),
    os.path.join('DR3', 'MTDT0', 'SI1994.WAV'),
    os.path.join('DR4', 'MTEB0', 'SX233.WAV'),
    os.path.join('DR3', 'FCMH0', 'SI824.WAV'),
    os.path.join('DR6', 'MRJR0', 'SX372.WAV'),
    os.path.join('DR2', 'MMDB1', 'SX275.WAV'),
    os.path.join('DR4', 'FDMS0', 'SX138.WAV'),
    os.path.join('DR4', 'FADG0', 'SX289.WAV'),
    os.path.join('DR4', 'FSEM0', 'SX388.WAV'),
    os.path.join('DR7', 'MERS0', 'SX209.WAV'),
    os.path.join('DR3', 'MRTK0', 'SX193.WAV'),
    os.path.join('DR4', 'MTEB0', 'SI1133.WAV'),
    os.path.join('DR3', 'FKMS0', 'SX50.WAV'),
    os.path.join('DR8', 'FJSJ0', 'SI854.WAV'),
    os.path.join('DR4', 'FADG0', 'SX379.WAV'),
    os.path.join('DR3', 'MCSH0', 'SX379.WAV'),
    os.path.join('DR7', 'MERS0', 'SX389.WAV'),
    os.path.join('DR1', 'MJSW0', 'SX380.WAV'),
    os.path.join('DR3', 'MWJG0', 'SI1754.WAV'),
    os.path.join('DR4', 'MTEB0', 'SX323.WAV'),
    os.path.join('DR4', 'FEDW0', 'SX274.WAV'),
    os.path.join('DR4', 'FNMR0', 'SI2029.WAV'),
    os.path.join('DR5', 'FMAH0', 'SX29.WAV'),
    os.path.join('DR3', 'MTHC0', 'SI1015.WAV'),
    os.path.join('DR3', 'MRTK0', 'SX103.WAV'),
    os.path.join('DR6', 'MJFC0', 'SX223.WAV'),
    os.path.join('DR7', 'FMML0', 'SI2300.WAV'),
    os.path.join('DR5', 'MRWS1', 'SX320.WAV'),
    os.path.join('DR3', 'MMJR0', 'SI1648.WAV'),
    os.path.join('DR4', 'FEDW0', 'SI1653.WAV'),
    os.path.join('DR2', 'MMDB1', 'SX185.WAV'),
    os.path.join('DR4', 'FDMS0', 'SX48.WAV'),
    os.path.join('DR4', 'FSEM0', 'SI568.WAV'),
    os.path.join('DR1', 'FDAC1', 'SX34.WAV'),
    os.path.join('DR3', 'MMJR0', 'SI2166.WAV'),
    os.path.join('DR4', 'FGJD0', 'SX189.WAV'),
    os.path.join('DR4', 'MDLS0', 'SX278.WAV'),
    os.path.join('DR4', 'FSEM0', 'SI1198.WAV'),
    os.path.join('DR1', 'FAKS0', 'SI2203.WAV'),
    os.path.join('DR3', 'MTAA0', 'SX205.WAV'),
    os.path.join('DR4', 'FSEM0', 'SX298.WAV'),
    os.path.join('DR6', 'FDRW0', 'SI1423.WAV'),
    os.path.join('DR4', 'MDLS0', 'SX368.WAV'),
    os.path.join('DR3', 'MRTK0', 'SX13.WAV'),
    os.path.join('DR4', 'FEDW0', 'SI1714.WAV'),
    os.path.join('DR3', 'MMWH0', 'SX369.WAV'),
    os.path.join('DR4', 'MTEB0', 'SI2064.WAV'),
    os.path.join('DR3', 'MTHC0', 'SX295.WAV'),
    os.path.join('DR1', 'FJEM0', 'SI1264.WAV'),
    os.path.join('DR4', 'MBNS0', 'SI1850.WAV'),
    os.path.join('DR3', 'MTAA0', 'SX295.WAV'),
    os.path.join('DR4', 'FADG0', 'SX109.WAV'),
    os.path.join('DR3', 'MTDT0', 'SI994.WAV'),
    os.path.join('DR1', 'FJEM0', 'SX4.WAV'),
    os.path.join('DR4', 'FREW0', 'SX110.WAV'),
    os.path.join('DR3', 'FKMS0', 'SI860.WAV'),
    os.path.join('DR6', 'MJFC0', 'SI1033.WAV'),
    os.path.join('DR3', 'MMJR0', 'SX298.WAV'),
    os.path.join('DR7', 'MRJM4', 'SI859.WAV'),
    os.path.join('DR4', 'FNMR0', 'SX139.WAV'),
    os.path.join('DR4', 'FREW0', 'SX380.WAV'),
    os.path.join('DR1', 'MJSW0', 'SX200.WAV'),
    os.path.join('DR5', 'FMAH0', 'SX389.WAV'),
    os.path.join('DR5', 'MRWS1', 'SI500.WAV'),
    os.path.join('DR3', 'MBWM0', 'SX134.WAV'),
    os.path.join('DR3', 'MBDG0', 'SI1463.WAV'),
    os.path.join('DR4', 'FEDW0', 'SX4.WAV'),
    os.path.join('DR6', 'MJFC0', 'SX43.WAV'),
    os.path.join('DR3', 'MCSH0', 'SX289.WAV'),
    os.path.join('DR1', 'FJEM0', 'SX274.WAV'),
    os.path.join('DR3', 'FKMS0', 'SX140.WAV'),
    os.path.join('DR3', 'FKMS0', 'SI2120.WAV'),
    os.path.join('DR7', 'FMML0', 'SX230.WAV'),
    os.path.join('DR3', 'MBWM0', 'SX404.WAV'),
    os.path.join('DR6', 'MJFC0', 'SX313.WAV'),
    os.path.join('DR3', 'MBWM0', 'SI674.WAV'),
    os.path.join('DR7', 'MRJM4', 'SI2119.WAV'),
    os.path.join('DR1', 'MJSW0', 'SI2270.WAV'),
    os.path.join('DR7', 'MDVC0', 'SX36.WAV'),
    os.path.join('DR4', 'FREW0', 'SI1280.WAV'),
    os.path.join('DR3', 'MMJR0', 'SX388.WAV'),
    os.path.join('DR2', 'MMDM2', 'SI1555.WAV'),
    os.path.join('DR3', 'MTAA0', 'SX25.WAV'),
    os.path.join('DR3', 'MMWH0', 'SI1089.WAV'),
    os.path.join('DR4', 'MROA0', 'SX227.WAV'),
    os.path.join('DR3', 'MBWM0', 'SX314.WAV'),
    os.path.join('DR3', 'MTDT0', 'SX4.WAV'),
    os.path.join('DR3', 'MMWH0', 'SX9.WAV'),
    os.path.join('DR1', 'FAKS0', 'SX43.WAV'),
    os.path.join('DR4', 'FSEM0', 'SI1828.WAV'),
    os.path.join('DR4', 'FNMR0', 'SX229.WAV'),
    os.path.join('DR4', 'MTEB0', 'SI503.WAV'),
    os.path.join('DR7', 'MRJM4', 'SX139.WAV'),
    os.path.join('DR3', 'FCMH0', 'SX104.WAV'),
    os.path.join('DR3', 'FCMH0', 'SI2084.WAV'),
    os.path.join('DR2', 'MMDM2', 'SX282.WAV'),
    os.path.join('DR7', 'FMML0', 'SX410.WAV'),
    os.path.join('DR6', 'FDRW0', 'SI653.WAV'),
    os.path.join('DR4', 'MDLS0', 'SX98.WAV'),
    os.path.join('DR5', 'FCAL1', 'SX143.WAV'),
    os.path.join('DR7', 'MRCS0', 'SI593.WAV'),
    os.path.join('DR3', 'MMJR0', 'SX28.WAV'),
    os.path.join('DR1', 'MJSW0', 'SI1010.WAV'),
    os.path.join('DR2', 'MMDM2', 'SI1452.WAV'),
    os.path.join('DR4', 'FGJD0', 'SI549.WAV'),
    os.path.join('DR7', 'MRJM4', 'SI1489.WAV'),
    os.path.join('DR4', 'FJMG0', 'SI551.WAV'),
    os.path.join('DR1', 'FJEM0', 'SX94.WAV'),
    os.path.join('DR7', 'MDLF0', 'SX233.WAV'),
    os.path.join('DR5', 'MRWS1', 'SI1130.WAV'),
    os.path.join('DR3', 'MBWM0', 'SX44.WAV'),
    os.path.join('DR5', 'FCAL1', 'SI1403.WAV'),
    os.path.join('DR7', 'MRJM4', 'SX229.WAV'),
    os.path.join('DR3', 'MTHC0', 'SX25.WAV'),
    os.path.join('DR1', 'FDAC1', 'SX394.WAV'),
    os.path.join('DR5', 'FCAL1', 'SX413.WAV'),
    os.path.join('DR2', 'MMDM2', 'SX12.WAV'),
    os.path.join('DR3', 'FKMS0', 'SX230.WAV'),
    os.path.join('DR4', 'MDLS0', 'SI1628.WAV'),
    os.path.join('DR3', 'MBDG0', 'SX113.WAV'),
    os.path.join('DR7', 'MDVC0', 'SX126.WAV'),
    os.path.join('DR5', 'FMAH0', 'SX299.WAV'),
    os.path.join('DR3', 'MGLB0', 'SX94.WAV'),
    os.path.join('DR6', 'FDRW0', 'SX203.WAV'),
    os.path.join('DR1', 'FAKS0', 'SX223.WAV'),
    os.path.join('DR3', 'MGJF0', 'SX101.WAV'),
    os.path.join('DR4', 'MTEB0', 'SX143.WAV'),
    os.path.join('DR3', 'MMWH0', 'SX189.WAV'),
    os.path.join('DR4', 'MDLS0', 'SX188.WAV'),
    os.path.join('DR2', 'MMDB1', 'SI1625.WAV'),
    os.path.join('DR4', 'MBNS0', 'SX320.WAV'),
    os.path.join('DR7', 'MERS0', 'SI1649.WAV'),
    os.path.join('DR2', 'MPDF0', 'SI912.WAV'),
    os.path.join('DR2', 'MGWT0', 'SX279.WAV'),
    os.path.join('DR3', 'MMWH0', 'SX279.WAV'),
    os.path.join('DR8', 'FJSJ0', 'SX314.WAV'),
    os.path.join('DR1', 'MREB0', 'SX115.WAV'),
    os.path.join('DR5', 'MRWS1', 'SX50.WAV'),
    os.path.join('DR3', 'MTDT0', 'SI2254.WAV'),
    os.path.join('DR3', 'MGLB0', 'SI2164.WAV'),
    os.path.join('DR3', 'MWJG0', 'SX314.WAV'),
    os.path.join('DR3', 'MGJF0', 'SX191.WAV'),
    os.path.join('DR7', 'MDVC0', 'SI2174.WAV'),
    os.path.join('DR3', 'MTHC0', 'SX385.WAV'),
    os.path.join('DR4', 'MROA0', 'SI1307.WAV'),
    os.path.join('DR4', 'MBNS0', 'SX230.WAV'),
    os.path.join('DR4', 'MROA0', 'SX317.WAV'),
    os.path.join('DR3', 'MWJG0', 'SX224.WAV'),
    os.path.join('DR2', 'MPDF0', 'SX372.WAV'),
    os.path.join('DR6', 'MJFC0', 'SI2293.WAV'),
    os.path.join('DR2', 'MGWT0', 'SI2169.WAV'),
    os.path.join('DR5', 'FCAL1', 'SX233.WAV'),
    os.path.join('DR3', 'MWJG0', 'SX404.WAV'),
    os.path.join('DR4', 'FREW0', 'SX200.WAV'),
    os.path.join('DR4', 'MDLS0', 'SX8.WAV'),
    os.path.join('DR7', 'MRJM4', 'SX49.WAV'),
    os.path.join('DR3', 'MMWH0', 'SI1301.WAV'),
    os.path.join('DR4', 'MBNS0', 'SI590.WAV'),
    os.path.join('DR4', 'FREW0', 'SI1030.WAV'),
    os.path.join('DR7', 'MDLF0', 'SI1583.WAV'),
    os.path.join('DR7', 'MERS0', 'SX119.WAV'),
    os.path.join('DR2', 'MPDF0', 'SX102.WAV'),
    os.path.join('DR3', 'FKMS0', 'SX410.WAV'),
    os.path.join('DR4', 'FGJD0', 'SX9.WAV'),
    os.path.join('DR2', 'MGWT0', 'SX9.WAV'),
    os.path.join('DR3', 'MBWM0', 'SI1304.WAV'),
    os.path.join('DR3', 'MGJF0', 'SI1901.WAV'),
    os.path.join('DR4', 'FREW0', 'SX20.WAV'),
    os.path.join('DR6', 'MRJR0', 'SX12.WAV'),
    os.path.join('DR5', 'FCAL1', 'SI2033.WAV'),
    os.path.join('DR8', 'FJSJ0', 'SI1484.WAV'),
    os.path.join('DR7', 'MERS0', 'SI1019.WAV'),
    os.path.join('DR4', 'FSEM0', 'SX118.WAV'),
    os.path.join('DR6', 'MJFC0', 'SI1663.WAV'),
    os.path.join('DR2', 'MMDB1', 'SX95.WAV'),
    os.path.join('DR4', 'FJMG0', 'SX281.WAV'),
    os.path.join('DR7', 'MRCS0', 'SX233.WAV'),
    os.path.join('DR7', 'MDVC0', 'SI2196.WAV'),
    os.path.join('DR3', 'MGLB0', 'SI1534.WAV'),
    os.path.join('DR3', 'MCSH0', 'SX19.WAV'),
    os.path.join('DR6', 'FDRW0', 'SX383.WAV'),
    os.path.join('DR3', 'MCSH0', 'SI919.WAV'),
    os.path.join('DR1', 'MJSW0', 'SX110.WAV'),
    os.path.join('DR4', 'FNMR0', 'SI1399.WAV'),
    os.path.join('DR3', 'MGJF0', 'SX11.WAV'),
    os.path.join('DR4', 'FGJD0', 'SX279.WAV'),
    os.path.join('DR6', 'MRJR0', 'SX102.WAV'),
    os.path.join('DR4', 'FADG0', 'SI649.WAV'),
    os.path.join('DR2', 'MPDF0', 'SX12.WAV'),
    os.path.join('DR4', 'MDLS0', 'SI2258.WAV'),
    os.path.join('DR2', 'MJAR0', 'SX98.WAV'),
    os.path.join('DR3', 'MRTK0', 'SI1093.WAV'),
    os.path.join('DR5', 'MRWS1', 'SI1496.WAV'),
    os.path.join('DR3', 'MMWH0', 'SI459.WAV'),
    os.path.join('DR3', 'MBDG0', 'SX293.WAV'),
    os.path.join('DR2', 'MMDB1', 'SX365.WAV'),
    os.path.join('DR7', 'MRJM4', 'SX319.WAV'),
    os.path.join('DR7', 'MDVC0', 'SX396.WAV'),
    os.path.join('DR3', 'MTDT0', 'SX364.WAV'),
    os.path.join('DR4', 'FGJD0', 'SI818.WAV'),
    os.path.join('DR4', 'FADG0', 'SI1279.WAV'),
    os.path.join('DR2', 'MJAR0', 'SX188.WAV'),
    os.path.join('DR4', 'FJMG0', 'SX371.WAV'),
    os.path.join('DR6', 'FDRW0', 'SX293.WAV'),
    os.path.join('DR3', 'MGJF0', 'SX371.WAV'),
    os.path.join('DR1', 'FDAC1', 'SX214.WAV'),
    os.path.join('DR7', 'MDLF0', 'SX413.WAV'),
    os.path.join('DR1', 'FJEM0', 'SI634.WAV'),
    os.path.join('DR1', 'FAKS0', 'SX313.WAV'),
    os.path.join('DR3', 'MCSH0', 'SX109.WAV'),
    os.path.join('DR4', 'FADG0', 'SX199.WAV'),
    os.path.join('DR4', 'FJMG0', 'SX11.WAV'),
    os.path.join('DR7', 'MERS0', 'SX299.WAV'),
    os.path.join('DR1', 'MREB0', 'SX25.WAV'),
    os.path.join('DR8', 'MAJC0', 'SI2095.WAV'),
    os.path.join('DR4', 'FDMS0', 'SX318.WAV'),
    os.path.join('DR3', 'MBDG0', 'SX203.WAV'),
    os.path.join('DR7', 'MDLF0', 'SX53.WAV'),
    os.path.join('DR4', 'FJMG0', 'SX191.WAV'),
    os.path.join('DR4', 'MBNS0', 'SX50.WAV'),
    os.path.join('DR5', 'FMAH0', 'SI1289.WAV'),
    os.path.join('DR3', 'MGLB0', 'SI904.WAV'),
    os.path.join('DR4', 'FJMG0', 'SI1811.WAV'),
    os.path.join('DR3', 'FKMS0', 'SX320.WAV'),
    os.path.join('DR4', 'FREW0', 'SI1910.WAV'),
    os.path.join('DR3', 'MTDT0', 'SX94.WAV'),
    os.path.join('DR5', 'FMAH0', 'SI659.WAV'),
    os.path.join('DR4', 'FJMG0', 'SX101.WAV'),
    os.path.join('DR2', 'MMDB1', 'SI995.WAV'),
    os.path.join('DR6', 'MJFC0', 'SX133.WAV'),
    os.path.join('DR6', 'MRJR0', 'SI2313.WAV'),
    os.path.join('DR7', 'FMML0', 'SX50.WAV'),
    os.path.join('DR4', 'FDMS0', 'SI1502.WAV'),
    os.path.join('DR3', 'MGJF0', 'SX281.WAV'),
    os.path.join('DR4', 'MROA0', 'SI677.WAV'),
    os.path.join('DR5', 'MRWS1', 'SX410.WAV'),
    os.path.join('DR3', 'MMJR0', 'SI2278.WAV'),
    os.path.join('DR2', 'MMDM2', 'SI2082.WAV'),
    os.path.join('DR3', 'MRTK0', 'SX373.WAV'),
    os.path.join('DR3', 'MTAA0', 'SX115.WAV'),
    os.path.join('DR6', 'MJFC0', 'SX403.WAV'),
    os.path.join('DR1', 'FDAC1', 'SI844.WAV'),
    os.path.join('DR8', 'MAJC0', 'SX385.WAV'),
    os.path.join('DR4', 'FNMR0', 'SI769.WAV'),
    os.path.join('DR3', 'MMJR0', 'SX208.WAV'),
    os.path.join('DR2', 'MGWT0', 'SX99.WAV'),
    os.path.join('DR4', 'FDMS0', 'SI1848.WAV'),
    os.path.join('DR3', 'MTHC0', 'SX115.WAV'),
    os.path.join('DR1', 'MJSW0', 'SX290.WAV'),
    os.path.join('DR8', 'MAJC0', 'SX115.WAV'),
    os.path.join('DR4', 'FEDW0', 'SX94.WAV'),
    os.path.join('DR1', 'FDAC1', 'SX304.WAV'),
    os.path.join('DR5', 'MRWS1', 'SX230.WAV'),
    os.path.join('DR3', 'MWJG0', 'SI1124.WAV'),
    os.path.join('DR4', 'FEDW0', 'SX184.WAV'),
    os.path.join('DR2', 'MPDF0', 'SX192.WAV'),
    os.path.join('DR3', 'MRTK0', 'SX283.WAV'),
    os.path.join('DR3', 'MBDG0', 'SX23.WAV'),
    os.path.join('DR5', 'FCAL1', 'SX323.WAV'),
    os.path.join('DR1', 'FJEM0', 'SX364.WAV'),
    os.path.join('DR4', 'MBNS0', 'SX140.WAV'),
    os.path.join('DR3', 'MTAA0', 'SI1285.WAV'),
    os.path.join('DR3', 'MWJG0', 'SX44.WAV'),
    os.path.join('DR8', 'FJSJ0', 'SI2114.WAV'),
    os.path.join('DR4', 'MBNS0', 'SX410.WAV'),
    os.path.join('DR4', 'FSEM0', 'SX28.WAV'),
    os.path.join('DR6', 'FDRW0', 'SI1283.WAV'),
    os.path.join('DR3', 'MTAA0', 'SI596.WAV'),
    os.path.join('DR3', 'MTAA0', 'SI1915.WAV'),
    os.path.join('DR1', 'FAKS0', 'SX403.WAV'),
    os.path.join('DR2', 'MJAR0', 'SI1988.WAV'),
    os.path.join('DR1', 'FDAC1', 'SI2104.WAV'),
    os.path.join('DR2', 'MJAR0', 'SX278.WAV'),
    os.path.join('DR5', 'FCAL1', 'SI773.WAV'),
    os.path.join('DR7', 'MDVC0', 'SI936.WAV'),
    os.path.join('DR3', 'MGLB0', 'SX274.WAV'),
    os.path.join('DR4', 'MBNS0', 'SI1220.WAV'),
    os.path.join('DR3', 'FCMH0', 'SX374.WAV'),
    os.path.join('DR5', 'FMAH0', 'SX119.WAV'),
    os.path.join('DR3', 'MRTK0', 'SI1750.WAV'),
    os.path.join('DR8', 'MAJC0', 'SI835.WAV'),
    os.path.join('DR4', 'FNMR0', 'SX409.WAV'),
    os.path.join('DR3', 'MGLB0', 'SX184.WAV'),
    os.path.join('DR6', 'FDRW0', 'SX23.WAV'),
    os.path.join('DR2', 'MGWT0', 'SX189.WAV'),
    os.path.join('DR4', 'FGJD0', 'SX369.WAV'),
    os.path.join('DR7', 'MERS0', 'SX29.WAV'),
    os.path.join('DR8', 'FJSJ0', 'SX44.WAV'),
    os.path.join('DR3', 'FCMH0', 'SX194.WAV'),
    os.path.join('DR4', 'FGJD0', 'SI1179.WAV'),
    os.path.join('DR4', 'MDLS0', 'SI998.WAV'),
    os.path.join('DR3', 'MWJG0', 'SI494.WAV'),
    os.path.join('DR3', 'MCSH0', 'SI1549.WAV'),
    os.path.join('DR1', 'FJEM0', 'SX184.WAV'),
    os.path.join('DR4', 'MROA0', 'SX47.WAV'),
    os.path.join('DR1', 'FAKS0', 'SI1573.WAV'),
    os.path.join('DR3', 'FCMH0', 'SX14.WAV'),
    os.path.join('DR3', 'FCMH0', 'SI1454.WAV'),
    os.path.join('DR7', 'MDLF0', 'SI2213.WAV'),
    os.path.join('DR2', 'MPDF0', 'SX282.WAV'),
    os.path.join('DR2', 'MJAR0', 'SI728.WAV'),
    os.path.join('DR2', 'MJAR0', 'SX368.WAV'),
    os.path.join('DR2', 'MMDB1', 'SI2255.WAV'),
    os.path.join('DR7', 'MDLF0', 'SX323.WAV'),
    os.path.join('DR3', 'MTHC0', 'SX205.WAV'),
    os.path.join('DR3', 'MTDT0', 'SX184.WAV'),
    os.path.join('DR1', 'MJSW0', 'SI1640.WAV'),
    os.path.join('DR4', 'FJMG0', 'SI1181.WAV'),
    os.path.join('DR3', 'MWJG0', 'SX134.WAV'),
    os.path.join('DR1', 'FAKS0', 'SI943.WAV'),
    os.path.join('DR3', 'MBWM0', 'SI1934.WAV'),
    os.path.join('DR8', 'FJSJ0', 'SX224.WAV'),
    os.path.join('DR7', 'FMML0', 'SI1040.WAV'),
    os.path.join('DR3', 'MBDG0', 'SI833.WAV'),
    os.path.join('DR3', 'MTAA0', 'SX385.WAV'),
    os.path.join('DR2', 'MMDM2', 'SX192.WAV'),
    os.path.join('DR4', 'FADG0', 'SX19.WAV'),
    os.path.join('DR7', 'MRJM4', 'SX409.WAV'),
    os.path.join('DR3', 'FCMH0', 'SX284.WAV'),
    os.path.join('DR7', 'FMML0', 'SX140.WAV'),
    os.path.join('DR3', 'MRTK0', 'SI1723.WAV'),
    os.path.join('DR6', 'MRJR0', 'SI1812.WAV'),
    os.path.join('DR3', 'FKMS0', 'SI1490.WAV'),
    os.path.join('DR1', 'MJSW0', 'SX20.WAV'),
    os.path.join('DR7', 'MERS0', 'SI497.WAV'),
    os.path.join('DR3', 'MMJR0', 'SX118.WAV'),
    os.path.join('DR3', 'MTHC0', 'SI1645.WAV'),
    os.path.join('DR3', 'MGLB0', 'SX4.WAV'),
    os.path.join('DR2', 'MMDM2', 'SX102.WAV'),
    os.path.join('DR4', 'FNMR0', 'SX319.WAV'),
    os.path.join('DR8', 'MAJC0', 'SX295.WAV'),
    os.path.join('DR4', 'FSEM0', 'SX208.WAV'),
    os.path.join('DR1', 'MREB0', 'SX385.WAV'),
    os.path.join('DR7', 'MRCS0', 'SX53.WAV'),
    os.path.join('DR7', 'MDVC0', 'SX216.WAV'),
    os.path.join('DR2', 'MMDM2', 'SX372.WAV'),
    os.path.join('DR6', 'MRJR0', 'SX282.WAV'),
    os.path.join('DR4', 'MROA0', 'SX137.WAV'),
    os.path.join('DR4', 'FDMS0', 'SX228.WAV'),
    os.path.join('DR5', 'MRWS1', 'SX140.WAV'),
    os.path.join('DR3', 'MTHC0', 'SI2275.WAV'),
    os.path.join('DR1', 'MREB0', 'SI745.WAV'),
    os.path.join('DR8', 'MAJC0', 'SX25.WAV'),
    os.path.join('DR7', 'MRCS0', 'SI1223.WAV'),
    os.path.join('DR7', 'MRCS0', 'SX323.WAV'),
    os.path.join('DR7', 'FMML0', 'SX320.WAV'),
    os.path.join('DR6', 'MRJR0', 'SI1182.WAV'),
    os.path.join('DR1', 'MREB0', 'SX205.WAV'),
    os.path.join('DR3', 'MMWH0', 'SX99.WAV'),
    os.path.join('DR4', 'MTEB0', 'SX413.WAV'),
    os.path.join('DR8', 'MAJC0', 'SI1946.WAV'),
    os.path.join('DR1', 'FDAC1', 'SI1474.WAV'),
    os.path.join('DR7', 'MDVC0', 'SX306.WAV'),
    os.path.join('DR3', 'MCSH0', 'SX199.WAV'),
    os.path.join('DR6', 'FDRW0', 'SX113.WAV'),
    os.path.join('DR2', 'MMDB1', 'SX5.WAV'),
    os.path.join('DR3', 'MCSH0', 'SI2179.WAV'),
    os.path.join('DR3', 'MTDT0', 'SX274.WAV'),
    os.path.join('DR2', 'MJAR0', 'SX8.WAV'),
    os.path.join('DR1', 'MREB0', 'SX295.WAV'),
    os.path.join('DR5', 'FMAH0', 'SI1919.WAV'),
    os.path.join('DR1', 'MREB0', 'SI2005.WAV'),
    os.path.join('DR3', 'MGJF0', 'SI641.WAV'),
    os.path.join('DR4', 'MROA0', 'SI1970.WAV'),
    os.path.join('DR7', 'MRCS0', 'SI1853.WAV'),
    os.path.join('DR4', 'FDMS0', 'SX408.WAV'),
    os.path.join('DR4', 'FREW0', 'SX290.WAV'),
    os.path.join('DR4', 'FADG0', 'SI1909.WAV'),
    os.path.join('DR5', 'FMAH0', 'SX209.WAV'),
    os.path.join('DR7', 'MDLF0', 'SX143.WAV'),
    os.path.join('DR5', 'FCAL1', 'SX53.WAV'),
    os.path.join('DR2', 'MPDF0', 'SI1542.WAV'),
    os.path.join('DR3', 'MBDG0', 'SX383.WAV'),
    os.path.join('DR7', 'FMML0', 'SI1670.WAV'),
    os.path.join('DR7', 'MRCS0', 'SX143.WAV'),
    os.path.join('DR1', 'FDAC1', 'SX124.WAV'),
    os.path.join('DR4', 'FNMR0', 'SX49.WAV'),
    os.path.join('DR8', 'MAJC0', 'SX205.WAV'),
    os.path.join('DR4', 'MTEB0', 'SX53.WAV'),
    os.path.join('DR4', 'FDMS0', 'SI1218.WAV'),
    os.path.join('DR2', 'MGWT0', 'SX369.WAV'),
    os.path.join('DR2', 'MGWT0', 'SI1539.WAV'),
    os.path.join('DR2', 'MJAR0', 'SI2247.WAV'),
    os.path.join('DR7', 'MRCS0', 'SX413.WAV'),
    os.path.join('DR4', 'MROA0', 'SX407.WAV'),
    os.path.join('DR2', 'MGWT0', 'SI909.WAV'),
    os.path.join('DR1', 'FJEM0', 'SI1894.WAV'),
    os.path.join('DR3', 'MBDG0', 'SI2093.WAV'),
    os.path.join('DR7', 'MDLF0', 'SI953.WAV'),
    os.path.join('DR3', 'MGJF0', 'SI776.WAV'),
    os.path.join('DR6', 'MRJR0', 'SX192.WAV'),
    os.path.join('DR1', 'MREB0', 'SI1375.WAV'),
    os.path.join('DR4', 'FEDW0', 'SX364.WAV'),
    os.path.join('DR4', 'FEDW0', 'SI1084.WAV'),
    os.path.join('DR4', 'FGJD0', 'SX99.WAV'),
    os.path.join('DR8', 'FJSJ0', 'SX404.WAV'),
    os.path.join('DR8', 'FJSJ0', 'SX134.WAV'),
    os.path.join('DR3', 'MBWM0', 'SX224.WAV'),
    os.path.join('DR3', 'MGLB0', 'SX364.WAV'),
]
