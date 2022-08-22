
print("======================= 2. load_model 출력 ======================")
model2 = load_model('./_save/keras26_4_save_MCP.h5')
loss2 = model2.evaluate(x_test,y_test)
print('loss2 : ', loss2)

y_predict2 = model2.predict(x_test)

r2 = r2_score(y_test,y_predict2) 
print('r2스코어 : ', r2)

# print("====================== 3. mcp 출력 ============================")
# model3 = load_model('./_ModelCheckPoint/keras26_3_MCP.hdf5')
# loss3 = model3.evaluate(x_test,y_test)
# print('loss3 : ', loss3)

# y_predict3 = model3.predict(x_test)

# r2 = r2_score(y_test,y_predict3) 
# print('r2스코어 : ', r2)