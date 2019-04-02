# AirlineOntimePerformance

1) load the data into a Kafka cluster under the relevant topics.

a)Airports 
cat /home/shadoop/tmp/flightdata/ airports.csv | /usr/local/kafka/bin/kafka-console-producer.sh --broker-list localhost:9092 --topic Airports
sudo /usr/local/kafka/bin/kafka-console-producer.sh --broker-list localhost:9092 --topic Airports < airports.csv
b)Carriers 
cat /home/shadoop/tmp/flightdata/carriers.csv | /usr/local/kafka/bin/kafka-console-producer.sh --broker-list localhost:9092 --topic Carriers
c)plane-data
cat /home/shadoop/tmp/flightdata/plane-date.csv | /usr/local/kafka/bin/kafka-console-producer.sh --broker-list localhost:9092 --topic plane-date

2) Download the stats created for year 2008 & 2007.

http://stat-computing.org/dataexpo/2009/the-data.html
hdfs dfs -put /home/shadoop/tmp/flightdata/2007.csv  /emirates/data/raw/
hdfs dfs -put /home/shadoop/tmp/flightdata/2008.csv  /emirates/data/raw/

3) Use Apache Flume to consume messages from Airports & Planedate Kafka Topic to HDFS Raw folder

Start all services in hadoop.
Start kafka server.
In the flume conf file mention kafka topic to write in HDFS.
And Mention HDFS Path.
Start the flume agent 
flume-ng agent -n flume1 -c conf -f /home/shadoop/apache-flume-1.6.0-bin/conf/flumeKafka.conf -Dflume.root.logger=INFO,console

4)flume will read messages from kafka topic it will write and generate tiny file in the HDFS path.

For merging tiny files in single file we can use hdfs merge command or we can increase roll size or roll count in flume conf file to generate single file.
hdfs merge command will generate blank lines while merging in single file.
So we can use below command to remove blank lines in file.
sed -i '/^$/d' flightplanedata.csv
hdfs merge command will merge files in local file system along with crc file which is hidden.
Again when we copy from local file sytem in hadoop path it gives check sum error if file size changes.
To avoid error, remove crc file from local directory and copy from local file system to hadoop path.

5)to consume messages from kafka  copy to hdfs directory in scala

Start zookeeper and kafka server consume messages
sudo  /usr/local/kafka/bin/zookeeper-server-start.sh /usr/local/kafka/config/zookeeper.properties
sudo  /usr/local/kafka/bin/kafka-server-start.sh /usr/local/kafka/config/server.properties
please paste scala file to consume  messages from kafka.

6) For each message in the Airports & Planedate data from raw directory, append UUID and timestamp using Pig Latin.

Start pig by using below command.
pig
start Job history server
mr-jobhistory-daemon.sh --config $HADOOP_CONF start historyserver
register '/usr/local/pig-0.17.0/lib/piggybank.jar';
1   AA = Load 'hdfs://localhost:8020/emirates/data/raw/KafkaPlainData2/part-00000-d7f16147-a5f4-488a-923e-a260fd19e161-c000.csv' using org.apache.pig.piggybank.storage.CSVExcelStorage(',');
2   BBB= RANK AA;
3   C = FILTER BBB BY $0 >1; //to remove autogenerated value column header
4   STORE C into 'hdfs://localhost:8020/emirates/data/raw/KafkaPlainData2/pig_Outputc/' using PigStorage(',');
5   AA1 = Load 'hdfs://localhost:8020/emirates/data/raw/KafkaPlainData2/pig_Outputc/part-m-00000' using org.apache.pig.piggybank.storage.CSVExcelStorage(',') as (id,tailnum,corpType,manufacturer,issue_date,model,status,aircraft_type,engine_type,year);
6   CH = foreach AA1 generate tailnum,corpType,manufacturer,issue_date,model,status,aircraft_type,engine_type,year,CurrentTime() as toDay;
7   STORE CH into 'hdfs://localhost:8020/emirates/data/raw/KafkaPlainData2/testcomplete/clean' using PigStorage(',');

Please find attached scala file to get prediction of flight delays. 

I am applying XGBoost algoritham for the same problem to minimize error to get good accuracy.