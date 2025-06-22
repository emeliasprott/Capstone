-- ----------------------------------------------------------------------------+
-- capublic.sql - Create all the capublic tables in MySQL database.
--
--  ---When---  ---Who---  ------------------What---------------------
--  2009-02-18  Rudy-H.    Created script to add tables.
--  2010-04-05  Rudy-H.    Added field DAYS_31ST_IN_PRINT to table
--                         BILL_TBL.
--  2010-05-18  Rudy-H.    Added field MEMBER_ORDER to table
--                         BILL_DETAIL_VOTE_TBL.
--  2011-03-18  Rudy-H.    Added more fields, indexes and changed fields.
--                         See changes_diff.txt file for actual changes.
--  2012-05-09  Rudy-H.    Added new table veto_message_tbl.
-- 2025-03-08   Emelia-S.  Converted to PostgreSQL.
-- ----------------------------------------------------------------------------
CREATE DATABASE legislation_db
    WITH ENCODING='UTF8'
    LC_COLLATE='en_US.UTF-8'
    LC_CTYPE='en_US.UTF-8'
    TEMPLATE template0;
-- -------------------------------------
-- Tables
-- -------------------------------------
DROP TABLE IF EXISTS legislation_db.bill_analysis_tbl;
CREATE TABLE legislation_db.bill_analysis_tbl
(
    analysis_id NUMERIC(22, 0) NOT NULL,
    bill_id VARCHAR(20) NOT NULL,
    house VARCHAR(10) NULL,
    analysis_type VARCHAR(100) NULL,
    committee_code VARCHAR(6) NULL,
    committee_name VARCHAR(200) NULL,
    amendment_author VARCHAR(100) NULL,
    analysis_date TIMESTAMP NULL,
    amendment_date TIMESTAMP NULL,
    page_num NUMERIC(22, 0) NULL,
    source_doc BYTEA NULL,
    released_floor VARCHAR(10) NULL,
    active_flg VARCHAR(10) NULL DEFAULT 'Y',
    trans_uid VARCHAR(20) NULL,
    trans_update TIMESTAMP NULL,
    PRIMARY KEY (analysis_id)
);
CREATE INDEX bill_analysis_bill_id_idx ON legislation_db.bill_analysis_tbl (bill_id);


DROP TABLE IF EXISTS legislation_db.bill_detail_vote_tbl;
CREATE TABLE legislation_db.bill_detail_vote_tbl
(
    bill_id VARCHAR(20) NOT NULL,
    location_code VARCHAR(10) NOT NULL,
    legislator_name VARCHAR(50) NOT NULL,
    vote_date_time TIMESTAMP NOT NULL,
    vote_date_seq INT NOT NULL,
    vote_code VARCHAR(5) NULL,
    motion_id INT NOT NULL,
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL,
    member_order INT NULL,
    session_date TIMESTAMP NULL,
    speaker VARCHAR(3) NULL
);
CREATE INDEX author_votecode_idx ON legislation_db.bill_detail_vote_tbl (legislator_name, vote_code);
CREATE INDEX bill_detail_vote_id_idx ON legislation_db.bill_detail_vote_tbl (bill_id);
CREATE INDEX check_dup_detail_vote_idx ON legislation_db.bill_detail_vote_tbl (bill_id, vote_date_time, location_code, motion_id, legislator_name, vote_date_seq);

DROP TABLE IF EXISTS legislation_db.bill_history_tbl;
CREATE TABLE legislation_db.bill_history_tbl
(
    bill_id VARCHAR(20) NULL,
    bill_history_id DECIMAL(20,0) NULL,
    action_date TIMESTAMP NULL,
    action_ VARCHAR(2000) NULL,
    trans_uid VARCHAR(20) NULL,
    trans_update_dt TIMESTAMP NULL,
    action_sequence INT NULL,
    action_code VARCHAR(5) NULL,
    action_status VARCHAR(60) NULL,
    primary_location VARCHAR(60) NULL,
    secondary_location VARCHAR(60) NULL,
    ternary_location VARCHAR(60) NULL,
    end_status VARCHAR(60) NULL
);
CREATE INDEX bill_history_id_idx ON legislation_db.bill_history_tbl (bill_id);

DROP TABLE IF EXISTS legislation_db.bill_motion_tbl;
CREATE TABLE legislation_db.bill_motion_tbl
(
    motion_id DECIMAL(20,0) NOT NULL,
    motion_text VARCHAR(250) NULL,
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL,
    PRIMARY KEY (motion_id)
);

DROP TABLE IF EXISTS legislation_db.bill_summary_vote_tbl;
CREATE TABLE legislation_db.bill_summary_vote_tbl
(
    bill_id VARCHAR(20) NOT NULL,
    location_code VARCHAR(10) NOT NULL,
    vote_date_time TIMESTAMP NOT NULL,
    vote_date_seq INT NOT NULL,
    motion_id INT NOT NULL,
    ayes INT NULL,
    noes INT NULL,
    abstain INT NULL,
    vote_result VARCHAR(10) NULL,
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL,
    file_item_num VARCHAR(10) NULL,
    file_location VARCHAR(50) NULL,
    display_lines VARCHAR(2000) NULL,
    session_date TIMESTAMP NULL
);
CREATE INDEX bill_summary_vote_id_idx ON legislation_db.bill_summary_vote_tbl (bill_id);
CREATE INDEX bill_summary_vote_mo_idx ON legislation_db.bill_summary_vote_tbl (motion_id);
CREATE INDEX check_dup_summary_vote_idx ON legislation_db.bill_summary_vote_tbl (bill_id, motion_id, vote_date_time, vote_date_seq, location_code);

DROP TABLE IF EXISTS legislation_db.bill_tbl;
CREATE TABLE legislation_db.bill_tbl
(
    bill_id VARCHAR(19) NOT NULL,
    session_year VARCHAR(8) NOT NULL,
    session_num VARCHAR(2) NOT NULL,
    measure_type VARCHAR(4) NOT NULL,
    measure_num INT NOT NULL,
    measure_state VARCHAR(40) NOT NULL,
    chapter_year VARCHAR(4) NULL,
    chapter_type VARCHAR(10) NULL,
    chapter_session_num VARCHAR(2) NULL,
    chapter_num VARCHAR(10) NULL,
    latest_bill_version_id VARCHAR(30) NULL,
    active_flg VARCHAR(1) NULL DEFAULT 'Y',
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL,
    current_location VARCHAR(200) NULL,
    current_secondary_loc VARCHAR(60) NULL,
    current_house VARCHAR(60) NULL,
    current_status VARCHAR(60) NULL,
    days_31st_in_print TIMESTAMP NULL,
    PRIMARY KEY (bill_id)
);
CREATE INDEX bill_tbl_chapter_year_idx ON legislation_db.bill_tbl (chapter_year);
CREATE INDEX bill_tbl_measure_num_idx ON legislation_db.bill_tbl (measure_num);
CREATE INDEX bill_tbl_measure_type_idx ON legislation_db.bill_tbl (measure_type);
CREATE INDEX bill_tbl_session_idx ON legislation_db.bill_tbl (session_year);
CREATE INDEX bill_tbl__ltst_bill_ver_idx ON legislation_db.bill_tbl (latest_bill_version_id);

DROP TABLE IF EXISTS legislation_db.bill_version_authors_tbl;
CREATE TABLE legislation_db.bill_version_authors_tbl
(
    bill_version_id VARCHAR(30) NOT NULL,
    type VARCHAR(15) NOT NULL,
    house VARCHAR(100) NULL,
    name VARCHAR(100) NULL,
    contribution VARCHAR(100) NULL,
    committee_members VARCHAR(2000) NULL,
    active_flg VARCHAR(1) NULL DEFAULT 'Y',
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL,
    primary_author_flg VARCHAR(1) NULL DEFAULT 'N'
);
CREATE INDEX bill_version_auth_tbl_id_idx ON legislation_db.bill_version_authors_tbl (bill_version_id);
CREATE INDEX bill_version_auth_tbl_name_idx ON legislation_db.bill_version_authors_tbl (name);

DROP TABLE IF EXISTS legislation_db.bill_version_tbl;
CREATE TABLE legislation_db.bill_version_tbl
(
    bill_version_id VARCHAR(30) NOT NULL,
    bill_id VARCHAR(19) NOT NULL,
    version_num INT NOT NULL,
    bill_version_action_date TIMESTAMP NOT NULL,
    bill_version_action VARCHAR(100) NULL,
    request_num VARCHAR(10) NULL,
    subject VARCHAR(1000) NULL,
    vote_required VARCHAR(100) NULL,
    appropriation VARCHAR(3) NULL,
    fiscal_committee VARCHAR(3) NULL,
    local_program VARCHAR(3) NULL,
    substantive_changes VARCHAR(3) NULL,
    urgency VARCHAR(3) NULL,
    taxlevy VARCHAR(3) NULL,
    bill_xml TEXT NULL,
    active_flg VARCHAR(1) NULL DEFAULT 'Y',
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL,
    PRIMARY KEY (bill_version_id)
);
CREATE INDEX bill_version_tbl_bill_id_idx ON legislation_db.bill_version_tbl (bill_id);
CREATE INDEX bill_version_tbl_version_idx ON legislation_db.bill_version_tbl (version_num);

DROP TABLE IF EXISTS legislation_db.codes_tbl;
CREATE TABLE legislation_db.codes_tbl
(
    code VARCHAR(5) NULL,
    title VARCHAR(2000) NULL
);

DROP TABLE IF EXISTS legislation_db.committee_hearing_tbl;
CREATE TABLE legislation_db.committee_hearing_tbl
(
    bill_id VARCHAR(20) NULL,
    committee_type VARCHAR(2) NULL,
    committee_nr INT NULL,
    hearing_date TIMESTAMP NULL,
    location_code VARCHAR(10) NULL,
    trans_uid VARCHAR(30) NULL,
    trans_update_date TIMESTAMP NULL
);
CREATE INDEX committee_hear_bill_id_idx ON legislation_db.committee_hearing_tbl (bill_id);

DROP TABLE IF EXISTS legislation_db.daily_file_tbl;
CREATE TABLE legislation_db.daily_file_tbl
(
    bill_id VARCHAR(20) NULL,
    location_code VARCHAR(10) NULL,
    consent_calendar_code INT NULL,
    file_location VARCHAR(6) NULL,
    publication_date TIMESTAMP NULL,
    floor_manager VARCHAR(100) NULL,
    trans_uid VARCHAR(20) NULL,
    trans_update_date TIMESTAMP NULL,
    session_num VARCHAR(2) NULL,
    status VARCHAR(200) NULL
);
CREATE INDEX daily_file_pub_date_idx ON legislation_db.daily_file_tbl (publication_date);
CREATE INDEX daily_file_tbl_bill_id_idx ON legislation_db.daily_file_tbl (bill_id);

DROP TABLE IF EXISTS legislation_db.law_section_tbl;
CREATE TABLE legislation_db.law_section_tbl
(
    id VARCHAR(100) NULL,
    law_code VARCHAR(5) NULL,
    section_num VARCHAR(30) NULL,
    op_statues VARCHAR(10) NULL,
    op_chapter VARCHAR(10) NULL,
    op_section VARCHAR(20) NULL,
    effective_date TIMESTAMP NULL,
    law_section_version_id VARCHAR(100) NULL,
    division VARCHAR(100) NULL,
    title VARCHAR(100) NULL,
    part VARCHAR(100) NULL,
    chapter VARCHAR(100) NULL,
    article VARCHAR(100) NULL,
    history VARCHAR(1000) NULL,
    content_xml TEXT NULL,
    active_flg VARCHAR(1) NULL DEFAULT 'Y',
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL
);
CREATE INDEX law_section_tbl_pk ON legislation_db.law_section_tbl (id);
CREATE INDEX law_section_code_idx ON legislation_db.law_section_tbl (law_code);
CREATE INDEX law_section_id_idx ON legislation_db.law_section_tbl (law_section_version_id);
CREATE INDEX law_section_sect_idx ON legislation_db.law_section_tbl (section_num);

DROP TABLE IF EXISTS legislation_db.law_toc_sections_tbl;
CREATE TABLE legislation_db.law_toc_sections_tbl
(
    id VARCHAR(100) NULL,
    law_code VARCHAR(5) NULL,
    node_treepath VARCHAR(100) NULL,
    section_num VARCHAR(30) NULL,
    section_order NUMERIC(22, 0) NULL,
    title VARCHAR(400) NULL,
    op_statues VARCHAR(10) NULL,
    op_chapter VARCHAR(10) NULL,
    op_section VARCHAR(20) NULL,
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL,
    law_section_version_id VARCHAR(100) NULL,
    seq_num NUMERIC(22, 0) NULL
);
CREATE INDEX law_toc_sections_node_idx ON legislation_db.law_toc_sections_tbl (law_code, node_treepath);

DROP TABLE IF EXISTS legislation_db.law_toc_tbl;
CREATE TABLE legislation_db.law_toc_tbl
(
    law_code VARCHAR(5) NULL,
    division VARCHAR(100) NULL,
    title VARCHAR(100) NULL,
    part VARCHAR(100) NULL,
    chapter VARCHAR(100) NULL,
    article VARCHAR(100) NULL,
    heading VARCHAR(2000) NULL,
    active_flg VARCHAR(1) NULL DEFAULT 'Y',
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL,
    node_sequence NUMERIC(22, 0) NULL,
    node_level NUMERIC(22, 0) NULL,
    node_position NUMERIC(22, 0) NULL,
    node_treepath VARCHAR(100) NULL,
    contains_law_sections VARCHAR(1) NULL,
    history_note VARCHAR(350) NULL,
    op_statues VARCHAR(10) NULL,
    op_chapter VARCHAR(10) NULL,
    op_section VARCHAR(20) NULL
);
CREATE INDEX law_toc_article_idx ON legislation_db.law_toc_tbl (article);
CREATE INDEX law_toc_chapter_idx ON legislation_db.law_toc_tbl (chapter);
CREATE INDEX law_toc_code_idx ON legislation_db.law_toc_tbl (law_code);
CREATE INDEX law_toc_division_idx ON legislation_db.law_toc_tbl (division);
CREATE INDEX law_toc_part_idx ON legislation_db.law_toc_tbl (part);
CREATE INDEX law_toc_title_idx ON legislation_db.law_toc_tbl (title);

DROP TABLE IF EXISTS legislation_db.legislator_tbl;
CREATE TABLE legislation_db.legislator_tbl
(
    district VARCHAR(5) NOT NULL,
    session_year VARCHAR(8) NULL,
    legislator_name VARCHAR(30) NULL,
    house_type VARCHAR(1) NULL,
    author_name VARCHAR(200) NULL,
    first_name VARCHAR(30) NULL,
    last_name VARCHAR(30) NULL,
    middle_initial VARCHAR(1) NULL,
    name_suffix VARCHAR(12) NULL,
    name_title VARCHAR(34) NULL,
    web_name_title VARCHAR(34) NULL,
    party VARCHAR(4) NULL,
    active_flg VARCHAR(1) NOT NULL DEFAULT 'Y',
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL,
    active_legislator VARCHAR(1) NULL DEFAULT 'Y'
);

DROP TABLE IF EXISTS legislation_db.location_code_tbl;
CREATE TABLE legislation_db.location_code_tbl
(
    session_year VARCHAR(8) NULL,
    location_code VARCHAR(10) NOT NULL,
    location_type VARCHAR(1) NOT NULL,
    consent_calendar_code VARCHAR(2) NULL,
    description VARCHAR(60) NULL,
    long_description VARCHAR(200) NULL,
    active_flg VARCHAR(1) NULL DEFAULT 'Y',
    trans_uid VARCHAR(30) NULL,
    trans_update TIMESTAMP NULL,
    inactive_file_flg VARCHAR(1) NULL
);
CREATE INDEX location_code_tbl_pk1 ON legislation_db.location_code_tbl (location_code);
CREATE INDEX localtion_code_session_idx1 ON legislation_db.location_code_tbl (session_year);

DROP TABLE IF EXISTS legislation_db.veto_message_tbl;
CREATE TABLE legislation_db.veto_message_tbl
(
    bill_id VARCHAR(20) NULL,
    veto_date TIMESTAMP NULL,
    message TEXT NULL,
    trans_uid VARCHAR(20) NULL,
    trans_update TIMESTAMP NULL
);

DROP TABLE IF EXISTS legislation_db.committee_agenda_tbl;
CREATE TABLE legislation_db.committee_agenda_tbl
(
    committee_code VARCHAR(200) NULL,
    committee_desc VARCHAR(1000) NULL,
    agenda_date TIMESTAMP NULL,
    agenda_time VARCHAR(200) NULL,
    line1 VARCHAR(500) NULL,
    line2 VARCHAR(500) NULL,
    line3 VARCHAR(500) NULL,
    building_type VARCHAR(200) NULL,
    room_num VARCHAR(100) NULL
);

----------------------------  END OF CODE ------------------------
