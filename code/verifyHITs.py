####### Does not apply to cases that have been approved but not rewarded
##### 1, Having access key in file just for testing
##### 2,
import requests
import boto3
import sys
import re
MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
MTURK_LIVE = 'https://mturk-requester.us-east-1.amazonaws.com'

print ("HIT Helper")
live =  input("Press 1 for LIVE environment\nPress 0 for SANDBOX environment\n")
if (int(live) == 1):
    MTURK_ENDPOINT = MTURK_LIVE
    print ("---- CAUTION: You are using the LIVE environment ----\n")
else:
    MTURK_ENDPOINT = MTURK_SANDBOX
    print ("You are using the SANDBOX environment")


mturk = boto3.client('mturk',
    aws_access_key_id="AKIAJXAIE6J4OFHE6DSA",
    aws_secret_access_key="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",
    region_name='us-east-1',
    endpoint_url=MTURK_ENDPOINT
)

print("There is $" + mturk.get_account_balance()
      ['AvailableBalance'] + " in your account")

confirm = input("Please check if you have selected the correct environment. Type y to proceed\n")
if (confirm.lower() != "y"):
    sys.exit()

# Variables here
assignment_counter = 0
all_hits = []
target_hits = []
all_assignments = []

workersWithResume = []
workersNoResume = []
workersToReject = []

# Retrieve all HITs created by the account
print(">> Retrieving all HITs")
response = mturk.list_hits(
    MaxResults=100
)
all_hits = response['HITs']
print(">> " + str(len(all_hits)) + " hits retrieved")

# Iterate through all HITs, and choosing the HITs with the designated title
for hit in all_hits:
    if 'Academic survey about opinions on algorithm prediction on recidivism' in hit['Title']:
        target_hits.append(hit)

# Storing all assignments from the target HITs
for hit in target_hits:
    print ('>>Processing HIT: ' + hit['Title'])
    # has_next_token = True
    next_token = None

    response = mturk.list_assignments_for_hit(
        HITId=hit['HITId'],
        MaxResults=100,
        AssignmentStatuses=['Submitted'],
        # NextToken=next_token
    )
    all_assignments.extend(response['Assignments'])
    next_token = response.get('NextToken')

    while next_token is not None:

        response = mturk.list_assignments_for_hit(
            HITId=hit['HITId'],
            MaxResults=100,
            AssignmentStatuses=['Submitted'],
            NextToken = next_token
        )

        all_assignments.extend(response['Assignments'])
        next_token = response.get('NextToken')
        # has_next_token = True if next_token is not None else False


print(">>>> "+ str(len(all_assignments))+" assignments retrieved")

# sys.exit(0)

total_payment = 0
# Iterating through all assignment completed by the MTurk worker
for assignment in all_assignments:
    # Ignoring assignments that have either been approved or rejected if needed
    if assignment['AssignmentStatus'] != 'Submitted':
        print(">>>>>>Skip assignment that are approved/rejected")
        continue

    print(">>>>>>processing " + assignment['AssignmentId'] + " by worker "+assignment['WorkerId']+ " (has not been approved or rejected previously)")

    # Getting the code submitted by the MTurk worker using regular expression
    submitted_code = re.search('<FreeText>(.*)</FreeText>', assignment['Answer'])
    submitted_code = submitted_code.group(1)
    if submitted_code is not None:
        submitted_code = submitted_code.strip()
    print(submitted_code)

    # Assigning Qualification to worker
    qualification_response = mturk.associate_qualification_with_worker(
        QualificationTypeId='3B5O8SC7Q7BYCDOC355PMZUFK2MSLT',
        #For testing in sandbox
        #QualificationTypeId='34B45A9E4BQ899LEWTMQSPC7L1U5RF',
        WorkerId=assignment['WorkerId'],
        IntegerValue=1,
        SendNotification=False #Not notifying Turker
        )
    # print (qualification_response)
    # e.g., code example: U78QD18CB15W

    if submitted_code is None or not submitted_code.startswith('U78Q'):
        # Workers that did not complete the survey. Reject.
        workersToReject.append(assignment['WorkerId'])
        mturk.reject_assignment(
            AssignmentId=assignment['AssignmentId'],
            RequesterFeedback='Sorry you did not complete the survey correctly'
        )
        print(">>>>>>>>Rejected to " + submitted_code)
    else:
        pass_attention_check = True if submitted_code[-3] == '1' else False
        if pass_attention_check:
            # pass attention check, compute bonus
            score = 30 - int(submitted_code[5:7]) + 6
            workersWithResume.append(assignment['WorkerId'])
            approval_response = mturk.approve_assignment(
                AssignmentId=assignment['AssignmentId'],
                RequesterFeedback='Thank you!',
                OverrideRejection=False
            )

            if score != 0:
                mturk.send_bonus(
                    WorkerId=assignment['WorkerId'],
                    BonusAmount=str(round(score * 0.2, 2)),
                    # The Bonus amount is a US Dollar amount specified using a string (for example, "5" represents $5.00 USD and "101.42" represents $101.42 USD). Do not include currency symbols or currency codes.
                    AssignmentId=assignment['AssignmentId'],
                    Reason='Thank you finishing bonus questions!',
                    UniqueRequestToken='TaskFinished' + assignment['WorkerId']
                )
            print(">>>>>>>>Approved and Sent Bonus to: " + submitted_code)
            total_payment += 2 + score*0.2
        else:
            # failed attention check, no money
            workersNoResume.append(assignment['WorkerId'])
            mturk.approve_assignment(
                AssignmentId=assignment['AssignmentId'],
                RequesterFeedback='Thank you!',
                OverrideRejection=False
            )
            print(">>>>>>>>Approved and No Bonus to " + submitted_code)

print("****Total payment: {}".format(total_payment))

### Check status
for assignment in all_assignments:
    if assignment['AssignmentStatus'] == "submitted":
        print(assignment['WorkerId']+" worker not been approved")