CREATE OR REPLACE TABLE DS_GLOC_DEV_DB."00_REF"."00_BRANCH_AGENT_MAPPING" AS (
    SELECT
        "AgentId" AS AGT_ID,
        "BranchId" AS BRANCH_ID,
        "AgentName" AS AGT_NAME,
        "AgentStatus" AS AGT_STATUS,
        "AgentStatusTitle" AS AGT_STATUS_TITLE,
        "ContractStartDate" AS CONTRACT_START_DATE,
        "ContractEndDate" AS CONTRACT_END_DATE
    FROM "GG_DS_DB"."GLOC_LAPSE_AND_SURRENDER"."JC_DAGENTS"
);


-- THIS IS A COMMENT.
-- This is a new coment!
CREATE OR REPLACE TABLE DS_GLOC_DEV_DB."00_REF"."BRANCH_MANAGERS_V2" (
    UNIT_NUM VARCHAR,
    MANAGER_EMAIL VARCHAR
);


INSERT INTO DS_GLOC_DEV_DB."00_REF"."BRANCH_MANAGERS_V2"
VALUES  ('11600', 'richard.defreitas@myguardiangroup.com'),
        ('11601', 'clinton.corbin@myguardiangroup.com'),
        ('11602', 'david.gordon@myguardiangroup.com'),
        ('11700', 'jeffrey.mazely@myguardiangroup.com'),
        ('11702', 'gregg.mannette@myguardiangroup.com'),
        ('11703', 'mark.rodriguez@myguardiangroup.com'),
        ('15000', 'calvin.mendez@myguardiangroup.com'),
        ('15001', 'ian.williams@myguardiangroup.com'),
        ('15002', 'ricard.skerritt@myguardiangroup.com'),
        ('16000', 'sean.david@myguardiangroup.com'),
        ('16001', 'yannick.antoine@myguardiangroup.com'),
        ('16002', 'tara.lal-chan@myguardiangroup.com'),
        ('22500', 'lenox.barrow@myguardiangroup.com'),
        ('22505', 'maria.brewster@myguardiangroup.com'),
        ('22506', 'denise.richardson@myguardiangroup.com'),
        ('22509', 'nichelle.emmanuel@myguardiangroup.com'),
        ('23000', 'dexter.george@myguardiangroup.com'),
        ('23003', 'roger.brumant@myguardiangroup.com'),
        ('23006', 'anthony.davis@myguardiangroup.com'),
        ('23008', 'quincy.tyner@myguardiangroup.com'),
        ('26000', 'ricky.rampersad@myguardiangroup.com'),
        ('26001', 'kerwyn.ramroach@myguardiangroup.com'),
        ('27000', 'totaram.benasrie@myguardiangroup.com'),
        ('27002', 'anginee.ramberan@myguardiangroup.com'),
        ('32500', 'dale.mcleod@myguardiangroup.com'),
        ('32514', 'kendall.lowhar@myguardiangroup.com'),
        ('32518', 'lisa.pollonais@myguardiangroup.com'),
        ('32519', 'karl.crooks@myguardiangroup.com'),
        ('32520', 'juliette.jacob-felician@myguardiangroup.com'),
        ('33000', 'ricardo.smith@myguardiangroup.com'),
        ('33001', 'clementine.jardine@myguardiangroup.com'),
        ('33002', 'brent.jankie@myguardiangroup.com'),
        ('33003', 'nigel.sookbir@myguardiangroup.com'),
        ('33004', 'renison.ramkhelawan@myguardiangroup.com'),
        ('43000', 'lurtan.patterson@myguardiangroup.com'),
        ('43001', 'wendell.noel@myguardiangroup.com'),
        ('43003', 'ria.murray-payne@myguardiangroup.com'),
        ('51500', 'nathaniel.wiltshire@myguardiangroup.com'),
        ('51506', 'joel.quamina@myguardiangroup.com'),
        ('51507', 'maureen.joseph@myguardiangroup.com'),
        ('90000', 'ramlal.basdeo@myguardiangroup.com'),
        ('90002', 'larry.taichew@myguardiangroup.com'),
        ('90009', 'gail.singh@myguardiangroup.com'),
        ('90010', 'delon.ramsundar@myguardiangroup.com'),
        ('90011', 'raj.mohan@myguardiangroup.com'),
        ('90012', 'shivan.maharaj@myguardiangroup.com'),
        ('90013', 'nirmala.krishnanan@myguardiangroup.com'),
        ('90014', 'dinesh.dass@myguardiangroup.com'),
        ('92000', 'amado.marcano@myguardiangroup.com'),
        ('93000', 'david.cave@myguardiangroup.com'),
        ('94000', 'robin.taylor@myguardiangroup.com'),
        ('94001', 'deon.cummins@myguardiangroup.com');


CREATE OR REPLACE TABLE DS_GLOC_DEV_DB."00_REF"."00_MANAGER_IDS" AS (
    SELECT
       T1."AgentId" AS AGT_ID,
       T2.UNIT_NUM AS BRANCH_NO
    FROM DS_GLOC_DEV_DB."00_REF".BRANCH_AGENT_MAPPING AS T1
    INNER JOIN DS_GLOC_DEV_DB."00_REF".BRANCH_MANAGERS_V2 AS T2 ON
        T1."BranchId" = T2.UNIT_NUM
);


CREATE OR REPLACE TABLE DS_GLOC_DEV_DB."00_REF"."00_BRANCH_MANAGER_ASSOCIATIONS" AS (
    WITH BRANCH_UNIT_MANAGERS AS (
        SELECT UNIT_NUM               AS BRANCH_NUMBER,
               SUBSTR(UNIT_NUM, 0, 3) AS BRANCH_ID,
               MANAGER_EMAIL          AS BRANCH_MANAGER_EMAIL
        FROM DS_GLOC_DEV_DB."00_REF".BRANCH_MANAGERS_V2
        WHERE SUBSTR(UNIT_NUM, 4, 5) = '00'
    )
    SELECT
        UNIT_NUM AS BRANCH_NUMBER,
        T2.BRANCH_MANAGER_EMAIL AS BRANCH_MANAGER_EMAIL,
        IFF(SUBSTR(UNIT_NUM, 4, 5) = '00', NULL, MANAGER_EMAIL) AS UNIT_MANAGER_EMAIL
    FROM DS_GLOC_DEV_DB."00_REF".BRANCH_MANAGERS_V2 AS T1
    LEFT JOIN BRANCH_UNIT_MANAGERS AS T2 ON SUBSTR(T1.UNIT_NUM, 0, 3) = T2.BRANCH_ID
);
