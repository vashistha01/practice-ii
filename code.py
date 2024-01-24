def get_formatted_name(first, last, middle=None):
    """Generate a neatly formatted name."""
    if middle:
        full_name=f"{first} {middle} {last}"
    else:
       full_name= f"{first} {last}"
    return full_name.title()

def place_info(city, country, population=None):
    """Generate info about place."""
    if population:
       info=f"{city}, {country}, {population}"
    else:
        info=f"{city}, {country}"
    return info.title()


class AnonymousSurvey:
    """Collect anonymous answer to survey questions."""
    def __init__(self, question):
        """Store a question, and prepare to store responses."""
        self.question=question
        self.responses=[]
    
    def show_question(self):
        """Show the survey question."""
        print(self.question)

    def store_response(self,new_responses):
        """Store a single response to the survey."""
        self.responses.append(new_responses)
    
    def show_results(self):
        print(f"Survey Results:- ")
        for response in self.responses:
            print(f"\t-{response}")

class Employee:
    """A simple model to represent class 'Employee'."""
    def __init__(self,):
        pass
        
    def ask_ans(self):
        """Ask answers."""
        self.first=first
        self.last=last
        self.annual_salary=annual_salary
   
    def give_raise(self,raise_amount=5000):
        """Give raise to the employees by default 5000 if not given."""
        if raise_amount:
            employee=f"{self.first.title()} {self.last.title()} {self.annual_salary+raise_amount}"
        else:
            employee=f"{self.first.title()} {self.last.title()} {self.annual_salary+raise_amount}"
        return employee


import unittest
from testing import AnonymousSurvey

class TestAnonymousSurvey(unittest.TestCase):
    """Tests for 'class AnonymousSurvey'."""
    def setUp(self):
        """
        Create a survey and a set of reponses for use in all test methods.
        """
        question='Which language did you first learn to speak? '
        self.my_survey=AnonymousSurvey(question)
        self.responses=['English', 'Hindi', 'Sanskrit']


    def test_store_single_repsonse(self):
        """Store single response for the survey."""
        self.my_survey.store_response(self.responses[0])
        self.assertIn(self.responses[0], self.my_survey.responses)

    def test_store_three_response(self):
        """Test that three individual responses are stored properly."""
        for response in self.responses:
            self.my_survey.store_response(response)
        for response in self.responses:
            self.assertIn(response, self.my_survey.responses)

if __name__=='__main__':
    unittest.main()

import numpy as np
import sys
import pandas as pd
import matplotlib .pyplot as plt
student=[1,2,3,4]
df=pd.DataFrame(student,columns=['student'])
print(df)
print(df.isin(1,3))
if __name__ == '__main__':
    students={}
    for records in range(int(input())):
        name = input()
        score = float(input())
        if score in students:
            students[score].append(name)
        else:
            students[score]=[name]


sequence=[("UP", 5), ("DOWN", 3), ("LEFT", 3), ("RIGHT", 2)]
current_position=(0,0)
for direction, steps in sequence:
  if direction=='UP':
    current_position=(current_position[0],current_position[1]+steps)
  if direction=='DOWN':
    current_position=(current_position[0],current_position[1]-steps)
  if direction=='LEFT':
    current_position=(current_position[0]-steps,current_position[1])
  if direction=='RIGHT':
    current_position=(current_position[0]+steps,current_position[1])

distance= math.pow((current_position[0]**2+current_position[1]**2),0.5)
print(int(round(distance)))


n=int(input('Enter the number to check prime or not: '))
if n>=2:
  for i in range(2, int(n**0.5)+1):
    if n%i==0:
      print('NO')
      break
  else:
    print('YES')

n=int(input('Enter the range: '))
for num in range(1, n+1):
  sum=0
  temp=num
  while temp>0:
    digit=temp%10
    sum+=digit**3
    temp//=10
  if num==sum:
    print(num,end=' ')

for i in range(1,6):
  for j in range(i,0,-1):
    print('*',end=' ')
  print()
for i in range(5,1,-1):
  for j in range(i):
    print('*',end=' ')
  print()


for i in range(5,0,-1):
  for j in range(i,0,-1):
    print(j,end=' ')
  print()


for i in range(1,6):
  for j in range(i,0,-1):
    print(j,end=' ')
  print()

num=[1,2,3,4]
for i in range(1, len(num)+1):
  for combination in itertools.combinations(num, i):
    print(combination)


#hour=int(input('Enter hour hand: '))
#minute=int(input('Enter minute: '))
angle=abs(hour*30-minute*11/2)
if angle>180:
  angle=360-angle
#print(f'Your angle is {angle} degree.')


index_val=[('cse', 2019),('cse', 2020),('cse', 2021),('cse', 2022),('ece', 2019),('ece', 2020),('ece', 2021),('ece', 2022)]
a=pd.Series([1,2,3,4,5,6,7,8],index=index_val)
multiindex=pd.MultiIndex.from_tuples(index_val)
multiindex=pd.MultiIndex.from_product([['cse','ece'],[2019,2020,2021,2022]])
s=pd.Series([1,2,3,4,5,6,7,8],index=multiindex)
branch_df1=pd.DataFrame(
    [
        [1,2],
        [3,4],
        [5,6],
        [7,8],
        [9,10],
        [11,12],
        [13,14],
        [15,16],
    ],index=multiindex,columns=['avg_package','students']
)

branch_df2=pd.DataFrame(
    [
        [1,2,0,0],
        [3,4,0,0],
        [5,6,0,0],
        [7,8,0,0],
    ]
    ,index=[2019,2020,2021,2022],columns=pd.MultiIndex.from_product([['delhi','mumbai'],['avg_package','students']])
)
branch_df3=pd.DataFrame(
    [
        [1,2,0,0],
        [3,4,0,0],
        [5,6,0,0],
        [7,8,0,0],
        [9,10,0,0],
        [11,12,0,0],
        [13,14,0,0],
        [15,16,0,0],
    ]
    ,index=multiindex, columns=pd.MultiIndex.from_product([['delhi','mumbai'],['avg_package','students']])
)

try:
  print(5/0)
except ZeroDivisionError:
    print(f"You can't divide by 0!")

print(f"Give me two numbers and I'll divide them for you.  ")
print("Enter 'q' to quit. ")
while True:
    first_number=input(f"\nFirst number: ")
    if first_number=='q':
        break
    second_number=input(f"\nSecond number:  ")
    if second_number=='q':
        break
    try:
      answer=int(first_number)/int(second_number)
    except ZeroDivisionError:
        print("You can't divide by 0!")
    else:
        print(answer)


def count_words(file_name):
    """Count the approximate number of words in file."""
    try:
        with open(file_name, encoding='utf-8') as f:
             contents=f.read()
    except FileNotFoundError:
             pass
    else:
    #Count the approximate numbers of words in the file.
       words=contents.split()
       num_words=len(words)
       print(f"The file {file_name} has about {num_words} words. ")

title='Alice in wonderland'
print(title.split())
file_names=['alice.txt','programming.txt','gif.txt']
for file_name in file_names:
    count_words(file_name)

def find_file(file_name):
    """Finding files in floder."""
    try:
        with open(file_name, encoding='utf-8') as f:
            contents=f.read()
            
    except FileNotFoundError:
           pass
    else:
        print(contents)

file_names=['cats.txt','dogs.txt']
for file_name in file_names:
    find_file(file_name)

add=lambda x,y : (x+y)
print(add(3,2))


def appl(fx, value):
    return 6+ fx(value)

print(appl(lambda x: x*x*x, 2))

my_list=['banana','mango','apple']

my_list.extend(['litchi','guava'])
counter_list=list(enumerate(my_list,1))
print(counter_list)

marks=[12,91,8,27,89]
for index, mark in enumerate(marks):
    print(mark)
    if index==3:
        print('good')

lst1=['apple','pineapple']
lst2=['red','brown']
a=zip(lst1,lst2)
print(next(a))
print(next(a))

b=list(zip())
print(b)

dict1={'name':'jim','last_name':'black','age':76}
dict2={'name':'lee','last_name':'white','age':15}
dictionary=zip(dict1.items(),dict2.items())
for (i,j),(i2,j2)in dictionary:
    print(i,j)
    

import json
numbers=[3,5,7,11,13,15,17]
file_name='numbers.json'
with open(file_name,'w') as f:
    json.dump(numbers,f)

import os
print(os.path.join('usr', 'bin', 'spam'))
print(os.getcwd())
os.path.abspath('.')
print(os.path.isabs('.'))
print(os.path.isabs(os.path.abspath('.')))
print(os.path.relpath('C:\\Windows', 'C:\\'))
print(os.path.dirname('C:\\Windows'))
path='C://Windows\\System32\\calc.exe'
print(os.path.basename(path))
print(os.path.dirname(path))
calcFilePath='C://Windows\\System32\\calc.exe'
print(os.path.split(calcFilePath))
print(os.path.dirname(calcFilePath), os.path.basename(calcFilePath))
print(calcFilePath.split(os.path.sep))
print(os.path.getsize(calcFilePath))
#print(os.listdir('C://Windows//System32'))
print(os.path.getsize('C://Windows//System32'))
total_size=0
for file_name in os.listdir('C://Windows//System32'):
    total_size= total_size+os.path.getsize(os.path.join('C://Windows//System32', file_name))
print(total_size)
print(os.path.exists('C://Windows//System32'))
print(os.path.exists('C://Window'))
print(os.path.isfile('C://Windows//System32'))
print(os.path.isdir('C://Windows//System32'))
print(os.path.exists('D://'))
hello_file=open("C:\\Users\\Documents\\hello.txt.txt")
hello_content=hello_file.read()
print(hello_content)
sonnet_file=open("C:\\Users\\Documents\\sonnet.txt.txt")
lines=sonnet_file.readlines()
for line in lines:
    print(line)
import pprint
cats=[{'name' : 'Zophie', 'desc' : 'chubby'}, {'name' : 'Pokka', 'desc' : 'fluffy'}]
pprint.pformat(cats)
file_object=open('myCats.py', 'w')
file_object.write('cats = '+pprint.pformat(cats)+'\n')
file_object.close()
import myCats
print(myCats.cats)
print(myCats.cats[0]['name'])

import pandas as pd 
from pandas import Series, DataFrame
import numpy as np 
obj=pd.Series([4, 7, -5, 3])
print(obj)
print(obj.values)
print(obj.index)
obj2=pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2)
print(obj2.index)
print(obj2['a'])
obj2['d']=6
print(obj2[obj2>0])
print(obj2*2)
print(np.exp(obj2))
print('b' in obj2)
sdata={'Ohio':35000, 'Texas':71000, 'Oregon':16000, 'Utah':5000}
obj3=pd.Series(sdata)
print(obj3)
states=['California', 'Ohio', 'Texas', 'Oregon']
obj4=pd.Series(sdata, index= states)
print(obj4)
print(pd.notnull(obj4))
print(obj4.isnull())
print(pd.isnull(obj4))
print(obj3+obj4)
obj4.name='population'
obj4.index.name='state'
print(obj4)
print(obj)
obj.index=['Bob', 'Steve', 'Jeff', 'Ryan']
print(obj)
data={'state':['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
'year':[2000, 2001, 2002, 2001, 2002, 2003],
'pop':[1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame=pd.DataFrame(data)
print(frame)
print(pd.DataFrame(data, columns=['year', 'state', 'pop']))
frame2=pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three',
'four', 'five', 'six'])
print(frame2)
print(frame2.loc['three'])
frame2['debt']=16.5
print(frame2)
frame2['debt']=np.arange(6.)
print(frame2)
val=pd.Series([1.7, 1.5, -1.2], index=['five', 'four', 'two'])
frame2['debt']=val
print(frame2)
frame2['eastern']=frame2.state=='Ohio'
print(frame2)
del frame2['eastern']
print(frame2.columns)
pop={'Nevada' : {2001: 2.4, 2002: 2.9},
     'Ohio' : {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3=pd.DataFrame(pop, index=[2000, 2001, 2002])
print(frame3)
print(frame3.T)
print(pd.DataFrame(pop, index=[2001, 2002, 2003]))
pdata= {'Ohio': frame3['Ohio'][:-1],
        'Nevada': frame3['Nevada'][:2]}
print(frame3)
       
print(pd.DataFrame(pdata))
frame3.index.name='year'; frame3.columns.name='state'
print(frame3)
print(frame3.values)
print(frame2.values)
obj=pd.Series(range(3), index=['a', 'b', 'c'])
index=obj.index
print(index[1:])
labels=pd.Index(np.arange(3))
print(labels)
obj2=pd.Series([1.5, -2.5, 0], index=labels)
print(obj2)
print(obj2.index is labels)
print(frame3.columns)
print('Ohio' in frame3.columns)
print(2003 in frame3.index)
dub_labels=pd.Index(['foo', 'foo', 'bar', 'bar'])
print(dub_labels)
obj=pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
print(obj)
obj2=obj.reindex(['a', 'b', 'c', 'd', 'e'])
print(obj2)
obj3=pd.Series(['blue', 'purple', 'yellow'], index=[0,2,4])
print(obj3)
print(obj3.reindex(range(6), method= 'ffill'))
frame=pd.DataFrame(np.arange(9).reshape((3,3)),
      index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
print(frame)
frame=frame.reindex(['a', 'b', 'c', 'd'])
print(frame)
states=['Texas', 'Utah', 'California']
frame=frame.reindex(columns=states)
print(frame.loc[['a', 'b', 'c', 'd'], states])
obj=pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
print(obj)
new_obj=obj.drop('c')
print(new_obj)
data=pd.DataFrame(np.arange(16).reshape((4,4)),
     index=['Ohio', 'Calarado', 'Utah', 'Newyork'],
     columns=['one', 'two', 'three', 'four'])
print(data)
data.drop(['Calarado', 'Ohio'])
print(data.drop(['two', 'four'], axis='columns'))
print(obj)
obj=pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
print(obj)
print(obj['b'])
print(obj[1])
print(obj[2:4])
print(obj[['b', 'a', 'd']])
print(obj[[1,3]])
print(obj[obj<2])
print(obj['b':'c'])
obj['b':'c']=5
print(obj)
data=pd.DataFrame(np.arange(16).reshape((4,4)),
      index=['Ohio', 'Colorado', 'Utah', 'Newyork'],
      columns=['one', 'two', 'three', 'four'])
print(data)
print(data['two'])
print(data[['three', 'one']])
print(data[:2])
print(data[data['three']>5])
print(data>5)
data[data<5]=0
print(data)
print(data.loc['Colorado',['two', 'three']])
print(data.iloc[2,[3, 0, 1]])
print(data.iloc[2])
print(data.iloc[[1, 2], [3, 0, 1]])
print(data.loc[:'Utah', 'two'])
ser=pd.Series(np.arange(3.), index=['a', 'b', 'c'])
ser2=pd.Series(np.arange(3.))
print(ser)
print(ser2)
print(ser2.loc[:1])
print(ser.iloc[:1])
s1=pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2=pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'g', 'e', 'f'])
print(s1)
print(s2)
print(s1+s2)
df1=pd.DataFrame(np.arange(9.).reshape((3,3)), columns=list('bcd'),
index=['Ohio','Texas', 'Colorado'])
df2=pd.DataFrame(np.arange(12.).reshape((4,3)), columns=list('bde'),
index=['Utah', 'Ohio', 'Texas', 'Oregon'])
print(df1)
print(df2)
print(df1+df2)
df1=pd.DataFrame({'A' : [1, 2]})
df2=pd.DataFrame({'B' : [3, 4]})
print(df1)
print(df2)
print(df1+df2)
df1=pd.DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2=pd.DataFrame(np.arange(20.).reshape((4,5)), columns=list('abcde'))
df2.loc[1, 'b']=np.nan
print(df1)
print(df2)
print(df1+df2)
print(df1.add(df2, fill_value=0))
print(1/df1)
print(df1.reindex(columns=df2.columns, fill_value=0))
print(df1)
arr=np.arange(12.).reshape((3, 4))
print(arr)
print(arr[0])
print(arr-arr[0])
frame=pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series=frame.iloc[0]
print(series)
print(frame-series)
series2=pd.Series((range(3)), index=['b', 'e', 'f'])
print(frame+series2)
series3=frame['d']
print(frame)
print(series3)
print(frame.sub(series3, axis='index'))
frame=pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
index=['Utah', 'Ohio', 'Texas', 'Oregon'])
print(frame)
print(np.abs(frame))
f=lambda x: x.max()-x.min()
print(frame.apply(f))
print(frame.apply(f, axis='columns'))
def f(x):
      return pd.Series([x.min(), x.max()], index=['min', 'max'])
print(frame.apply(f))
format= lambda x: '%.2f' % x
print(frame.applymap(format))
print(frame['e'].map (format))
obj=pd.Series(range(4), index=['d', 'a', 'b', 'c'])
print(obj.index)
print(obj.sort_index())
frame=pd.DataFrame(np.arange(8).reshape((2,4)), index=['three', 'one'],
columns=['d', 'a', 'b', 'c'])
print(frame)
frame.sort_index()
print(frame.sort_index(axis=1))
print(frame.sort_index(axis=1, ascending=False))
obj=pd.Series([4, 7, -3, 2])
print(obj)
print(obj.sort_values())
obj=pd.Series([4, np.nan, 7, np.nan, -3, 2])
print(obj)
print(obj.sort_values())
frame=pd.DataFrame({'b':[4, 7, -3, 2], 'a':[0, 1, 0, 1]})
print(frame)
print(frame.sort_values(by='b'))
print(frame.sort_values(by='a'))
print(frame.sort_values(by=['a', 'b']))
obj=pd.Series([7, -5, 7, 4, 2, 0, 4])
print(obj)
print(obj.rank())
print(obj.rank(method='first'))
print(obj.rank(ascending=False, method='max'))
frame=pd.DataFrame({'b' : [4.3, 7, -3, 2], 'a' : [0, 1, 0, 1],
 'c' : [-2, 5, 8, -2.5]})
print(frame)
print(frame.rank(axis='columns'))
obj=pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
print(obj)
print(obj.index.is_unique)
print(obj['a'])
print(obj['c'])
df=pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'] )
print(df)
print(df.loc['b'])
df=pd.DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], 
[0.75, -1.3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
print(df)
print(df.sum())
print(df.sum(axis='columns', skipna=False))
print(df.idxmax())
print(df.idxmin())
print(df.cumsum())
print(df.describe())
obj=pd.Series(['a', 'a', 'b', 'c']*4)
print(obj.describe())
obj=pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques=obj.unique()
print(uniques)
print(pd.value_counts(obj.values, sort=False))
print(obj)
mask=obj.isin(['b', 'c'])
print(mask)
print(obj[mask])
to_match=pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])
unique_vals=pd.Series(['c', 'b', 'a'])
print(pd.Index(unique_vals).get_indexer(to_match))

#Transpose
def transpose(matrix):
    n = len(matrix)
    m = len(matrix[0])

    transposed = [[0 for j in range(n)] for i in range(m)]

    for i in range(n):
        for j in range(m):
            transposed[j][i] = matrix[i][j]

    return transposed

matrix = [[1,2,3],[4,5,6],[7,8,9]]
print(transpose(matrix))

# Buddy Strings
def buddyStrings(s, goal):
    if len(s) != len(goal):
        return False
    if s == goal:
        return len(set(s)) < len(s)
    diffs = [(a, b)for a, b in zip(s, goal) if a != b]
    return len(diffs) == 2 and diffs[0] == diffs[1][::-1]

s = "ab"
goal = "ba"

print(buddyStrings(s, goal))

# Uncommon words from two sentences.
def uncommonFromSentences(s1, s2):
    words1 = s1.split()
    words2 = s2.split()

    counts = {}
    for word in words1 + words2:
        counts[word] = counts.get(word, 0) + 1

    uncommon = [word for word in counts if counts[word] == 1]
    return uncommon

s1 = "this apple is sweet"
s2 = "this apple is sour"
print(uncommonFromSentences(s1, s2))

# Binary Search
def search(nums, target):
    low = 0
    high = len(nums) - 1

    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

nums = [-1,0,3,5,9,12]
target = 9

#Set Mismatch
def findErrorNums(nums):
    n = len(nums)
    
    duplicate = -1
    for num in nums:
        if nums[abs(num) - 1] < 0:
            duplicate = abs(num)
        else:
            nums[abs(num) - 1] *= -1
    
    missing = -1
    for i in range(n):
        if nums[i] > 0:
            missing = i + 1
    return [duplicate, missing]

nums = [1,2,2,4]
print(findErrorNums(nums))

# Reorder Routes
from collections import defaultdict
def minReorder(n, connections):
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append((v, 1))
        graph[v].append((u, 0))
            
    def dfs(node):
        nonlocal total
        visited.add(node)
            
        for neighbor, cost in graph[node]:
            if neighbor not in visited:
                total += cost
                dfs(neighbor)
    total = 0
    visited = set()
    dfs(0)
    return total

n = 6
connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
print(minReorder(n, connections))

# Detect Capitals
def detectCapitalUse(word):
    if word.isupper():
        return True
    elif word.islower():
        return True
    elif word.istitle():
        return True
    else:
        return False
print(detectCapitalUse("USA"))

# Check prefect number
def checkPerfectNumber(num):
    if num <= 1:
        return False
    div_sum = 1
    i = 2
        
    # Check divisors up to square root of num
    while i*i <= num:
        if num % i == 0:
            div_sum += i
            if i != num//i:
                div_sum += num//i
        i += 1
    # Check if num is a perfect number
    if div_sum == num:
        return True
    else:
        return False
print(checkPerfectNumber(28))

# Relative Ranks
def findRelativeRanks(scores):
    sorted_nums = sorted(scores, reverse = True)
    ranks = {}
    for i, j in enumerate(sorted_nums):
        if i == 0:
            ranks[j] = "Gold Medal"
        elif i == 1:
            ranks[j] = "Silver Medal"
        elif i == 2:
            ranks[j] = "Bronze Medal"
        else:
            ranks[j] = str(i + 1)
    return [ranks[j] for j in scores]

scores = [5,4,3,2,1]
print(findRelativeRanks(scores))

#Repeated Substring Patterns
def repeatedSubstringPattern(s):
    string = (s + s)[1:-1]
    return string.find(s) != -1

print(repeatedSubstringPattern("abcabcabcabc"))

# Counting Bits
def countBits(num):
    counter = [0]
    if num >= 1:
        while len(counter) <= num:
            counter = counter + [i + 1 for i in counter]
        return counter[:num+1]
    else:
        return 0
print(countBits(5))

# Valid Prefect Square
def isPerfectSquare(num):
    left = 1
    right = num

    while left < right:
        mid = (left + right) // 2
        if mid * mid == num:
            return True
        
        elif mid * mid < num:
            left = mid + 1
        
        elif mid * mid > num:
            right = mid
    return False

print(isPerfectSquare(49))

# First Unique Character
def firstUniqueChar(s):
    from collections import Counter
    count = Counter(s)
    for i , j in enumerate(s):
        if count[j] == 1:
            return i
    else:
        return -1

print(firstUniqueChar("Tae"))

# Assign Cookies
def findContentChildren(g, s):
    i = 0
    j = 0
    g = sorted(g)
    s = sorted(s)

    while i < len(g) and j < len(s):
        i += g[i] <= s[i]
        j = j + 1
    return i

g = [1,2,3]
s = [1,1]
print(findContentChildren(g, s))

# Hamming Distance
def hammingDistance(x, y):
    xor = x ^ y
    dist = 0
    while xor:
        dist += 1
        xor &= xor - 1
    return dist

x = 1
y = 4
print(hammingDistance(x, y))

# Max Consecutive Ones
def findMaxConsecutiveOnes(nums):
    max_count = 0
    count = 0
    for i in nums:
        if i == 1:
            count += 1
        else:
            max_count = max(max_count, count)
            count = 0
    return max(max_count, count)

nums = [1,1,0,1,1,1]
print(findMaxConsecutiveOnes(nums))

# Construct Rectangle
import math
def constructRectangle(area):
    w = int(math.sqrt(area))
    while area % w != 0:
        w -= 1
    return [area // w, w]

area = 24
print(constructRectangle(area))

# License Key Formatting
def licenseKeyFormatting(s, k):
    s = s.upper().replace('-', '')
    size = len(s)
    s = s[::-1]
    res = []
    for i in range(0, size, k):
        res.append(s[i:i+k])
    return '-'.join(res)[::-1]

# Number of segments in Python
def countSegments(s):
    count = len(s.split())
    return count

print(countSegments("Hey, my Suga."))

# Third Maximum Number
def thirdMax(nums):
    nums.sort(reverse = True)
    count = 1
    previous = nums[0]

    for i in range(len(nums)):
        if nums[i] != previous:
            count = count + 1
            previous = nums[i]
        if count == 3:
            return nums[i]
    return nums[0] 

nums = [10, 2, 4, 5, 6, 2, 9, 1, 7, 5, 4, 11, 30]
print(thirdMax(nums))

# FizzBuzz Problem
def fizzBuzz(n):
    output = []

    for i in range(1, n + 1):
        if (i % 3) == 0 and (i % 5) == 0:
            output.append("FizzBuzz")
        elif i % 3 == 0:
            output.append("Fizz")
        elif i % 5 == 0:
            output.append("Buzz")
        else:
            output.append(str(i))
    return output

print(fizzBuzz(15))

# Reverse String
def reverse_string(string):
    return string[::-1]

a = "sednem nwahs"
print(reverse_string(a))

# Powers of three
def isPowerOfThree(n):
    while (n != 1):
        if (n % 3 != 0):
            return False
        n = n // 3
    else:
        return True
print(isPowerOfThree(27))

# Move Zeroes
def moveZeroes(nums):
    zeroes = 0
    for i in range(len(nums)):
        if nums[i] > 0:
            nums[zeroes], nums[i] = nums[i], nums[zeroes]
            zeroes = zeroes + 1
    return nums

nums = [0, 1, 0, 3, 12]
print(moveZeroes(nums))

# Add digits
def addDigits(num):
    while num > 9:
        num = (num % 10) + (num // 10)
    return num

print(addDigits(38))

# Power of two
def isPowerOfTwo(n):
    while (n != 1):
        if (n % 2 != 0):
            return False
        n = n // 2
    else:
        return True
print(isPowerOfTwo(16))

# Find Duplicated Values
def find_duplicates(x):
    length = len(x)
    duplicates = []
    for i in range(length):
        n = i + 1
        for a in range(n, length):
            if x[i] == x[a] and x[i] not in duplicates:
                duplicates.append(x[i])
    return duplicates
names = ["Aman", "Akanksha", "Divyansha", "Devyansh", 
         "Aman", "Diksha", "Akanksha"]
print(find_duplicates(names))

# pascal's Triangle
def generate(numRows):
    triangle = [[1]]
    row = 0
    while numRows > len(triangle):
        row = row + 1
        triangle.append([1] * (row + 1))
        for i in range(1, row):
            triangle[row][i] = triangle[row - 1][i - 1] + triangle[row -1][i]
    return triangle

print(generate(5))

# Check Duplicates Values
def containsDuplicate(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    else:
        return False

nums = [1,2,3,1]
print(containsDuplicate(nums))

# Majority Elements
def majorityElement(nums):
    count = 0
    major_element = 0
    for i in nums:
        if count == 0:
            major_element = i
        if major_element == i:
            count = count + 1
        else:
            count = count - 1
    return major_element

nums = [2,2,1,1,1,2,2]
print(majorityElement(nums))

# Excel Sheet Column Title
def convertToTitle(n):
    title = ""
    while n:
        n = n - 1
        title = chr(n % 26 + 65) + title
        n = n // 26
    return title
print(convertToTitle(28))

# Single Number
def singleNumber(nums):
    count = 0
    for i in nums:
        count = count^i
    return count

nums = [4, 1, 2, 1, 2]
print(singleNumber(nums))

# Best time to buy and sell stocks
def maxProfit(prices):
    buy = 0
    sell = 1
    max_profit = 0
    while sell < len(prices):
        if prices[sell] > prices[buy]:
            profit = prices[sell] - prices[buy]
            max_profit = max(profit, max_profit)
        else:
            buy = sell
        sell = sell + 1
    return max_profit

prices = [7,1,5,3,6,4]
print(maxProfit(prices))

# Climbing Stairs
def climbStairs(num):
    a = 1
    b = 1
    n = num - 1
    for i in range(n):
        c = a
        a = a + b
        b = c
    return a

print(climbStairs(4))

# Find Missing Numbers
def findMissingNumbers(n):
    numbers = set(n)
    length = len(n)
    output = []
    for i in range(1, n[-1]):
        if i not in numbers:
            output.append(i)
    return output
    
listOfNumbers = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16]
print(findMissingNumbers(listOfNumbers))

# Two Sums Problem
def twosum(nums, target):
    length = len(nums)
    for i in range(length):
        for j in range(i + 1, length):
            if nums[i] + nums[j] == target:
                return [i, j]

n = [3, 1, 1, 2]
t = 5
print(twosum(n, t))

# Solving Plus One Problem
def plusOne(digits):
    n = len(digits) - 1
    while digits[n] == 9:
        digits[n] = 0
        n = n - 1
    if n < 0:
        digits = [1] + digits
    else:
        digits[n] = digits[n] + 1
    return digits

digits = [1, 2, 5, 7]
print(plusOne(digits))

# Remove duplicates from sorted arrays
def removeDuplicate(items):
    list1 = []
    for i in items:
        if i not in list1:
            list1.append(i)
    return list1

nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
print(removeDuplicate(nums))

# Square Root
def mySqrt(x):
    left = 1
    right = x
    mid = 0
    while (left <= right):
        mid = (left + right) // 2
        if mid * mid == x:
            return mid
        elif mid * mid > x:
            right = mid - 1
        else:
            left = mid + 1
            sqrt = mid
    return sqrt

print(mySqrt(25))

# Merge two sorted lists
def merge2Lists(i, j):
    mergedlist = []
    while (i and j):
        if (i[0] <= j[0]):
            item = i.pop(0)
            mergedlist.append(item)
        else:
            item = j.pop(0)
            mergedlist.append(item)
    mergedlist.extend(i if i else j)
    return mergedlist

list1 = [1, 3, 5, 7, 9]
list2 = [2, 4, 6, 8, 10]

print(merge2Lists(list1, list2))

# Longest Common Prefix
strs = ["flower", "flow", "flight"]

def longestCommonPrefix(strs):
    l = len(strs[0])
    for i in range(1, len(strs)):
        length = min(l, len(strs[i]))
        while length > 0 and strs[0][0:length] != strs[i][0:length]:
            length = length - 1
            if length == 0:
                return 0
    return strs[0][0:length]

print(longestCommonPrefix(strs))

# Group Elements of Same indices
inputLists = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
outputLists = []
index = 0

for i in range(len(inputLists[0])):
    outputLists.append([])
    for j in range(len(inputLists)):
        outputLists[index].append(inputLists[j][ index])
    index = index + 1
a, b, c = output

# Group Anagram
from collections import defaultdict

def group_anagrams(a):
    dfdict = defaultdict(list)
    for i in a:
        sorted_i = " ".join(sorted(i))
        dfdict[sorted_i].append(i)
    return dfdict.values()

words = ["tea", "eat", "bat", "ate", "arc", "car"]
print(group_anagrams(words))

# Calculating execution timing
from time import time
start = time()

# Python program to create acronyms
word = "Artificial Intelligence"
text = word.split()
a = " "
for i in text:
    a = a+str(i[0]).upper()
print(a)

end = time()
execution_time = end - start
print("Execution Time : ", execution_time)

# Most frequent word in file
words = []
with open("a.txt", "r") as f:
    for line in f:
        words.extend(line.split())

from collections import Counter
counts = Counter(words)
top5 = counts.most_common(5)
print(top5)

[('the', 5), ('you', 5), ('Python', 4), ('is', 4), ('of', 3)]

# Index of the maximum value
def maximum(x):
    maximum_index = 0
    current_index = 1
    while current_index < len(x):
        if x[current_index] > x[maximum_index]:
            maximum_index = current_index
        current_index = current_index + 1
    return maximum_index
a = [23, 76, 45, 20, 70, 65, 15, 54]
print(maximum(a))

# Index of the minimum value
def minimum(x):
    minimum_index = 0
    current_index = 1
    while current_index < len(x):
        if x[current_index] < x[minimum_index]:
            minimum_index = current_index
        current_index = current_index + 1
    return minimum_index
a = [23, 76, 45, 20, 70, 65, 15, 54]
print(minimum(a))

# Calculate distance between two locations
import numpy as np
# Set the earth's radius (in kilometers)
r = 6371

# Convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi/180)

# Function to calculate the distance between two points 
# using the haversine formula
def distcalculate(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon = deg_to_rad(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return r * c
print(distcalculate(22.745049, 75.892471, 22.765049, 75.912471))




