      01 MQ-PAYLOAD.
         05 MQMD-FORMAT     PIC X(8).
         05 MQMD-PRIORITY   PIC 9(3).
         05 MQMD-ENCODING   PIC 9(3).
         05 MQMD-CHARSET    PIC 9(5).
         05 APP-ID          PIC X(12).
         05 PAYLOAD-LEN     PIC 9(5)   USAGE COMP.
         05 PAYLOAD-DATA    PIC X(32).
