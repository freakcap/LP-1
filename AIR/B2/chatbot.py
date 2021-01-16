from nltk.chat.util import Chat, reflections

pairs = [
            [
                r"my name is(.*)",
                ["Hello %1, how are you today?",]
            ],
            [
                r"what is your name?",
                ["My name is Chatbot and I will help you with your finacial queries today.",]
            ],
            [
                r"where to put(.*)money",
                ["Basically there are many options to invest- 1.Regional and 2.Stocks.\nIn which section would you like to invest?",]
            ],
            [
                r"Regional(.*)",
                ["There are many- SBI,HSBC,DB. Which bank would you like to go for?",]
            ],
            [
                r"SBI(.*)",
                ["SBI offers 10 percent Interest.",]
            ],
            [
                r"DB(.*)",
                ["DB offers 09 percent Interest.",]
            ],
            [
                r"HSBC(.*)",
                ["HSBC offers 11 percent Interest.",]
            ],
            [
                r"(.*)Stocks(.*)",
                ["We have 2 companies to offer: 1. AAA 2. BBB.\n choose any one to know more.\n",]
            ],
            [
                r"AAA(.*)",
                ["The company AAA has a ROI = 10 percent",]
            ],
            [
                r"BBB(.*)",
                ["The company BBB has a ROI = 12 percent",]
            ],
            [
                r"hi|hey|hello(.*)",
                ["Hello", "Hey there",]
            ],
            [
                r"quit",
                ["Signing out, see you again ^_^",]
            ],
        ]

def chatbot():
    
    print("Booting up...\nHey I am a simple ChatBot (without any AI), only using NLTK library.\n Please type in lower case what you want to ask me.\nPress Q to exit")
    chat = Chat(pairs, reflections)
    chat.converse()

if __name__ == "__main__":
    chatbot()
