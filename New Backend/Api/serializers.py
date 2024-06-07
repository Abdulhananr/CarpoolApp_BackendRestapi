# from datetime import timezone
from django.db import models
from django.utils.translation import check_for_language
from rest_framework import fields, serializers
from .models import Contact,Carpool,PrivateRoom,Voucher,History_Client,History_Driver,Carrypic
from .models import Customer,Client_location,Driver_location,Carpool_request,DCarpool_request,Final_Carpool
class Contactserializer(serializers.ModelSerializer):
    class Meta:
        model=Contact
        fields=['id','name','email','phone','desc']
class Carrypicserializer(serializers.ModelSerializer):
    class Meta:
        model=Carrypic
        fields=['id','pic']
class PrivateRoomserializer(serializers.ModelSerializer):
    class Meta:
        model=PrivateRoom
        fields=['id','sender','msg','reciver','pic','name','status']

class Driverserializer(serializers.ModelSerializer):
    class Meta:
        model=Driver_location
        fields=['id','did','lat','long']



class Clientserializer(serializers.ModelSerializer):
    class Meta:
        model=Client_location
        fields=['id','cid','lat','long']


#Show A request for accept and reject 
class Carpoolserializer(serializers.ModelSerializer):
    class Meta:
        model=Carpool
        fields=['id','lat','long','des_lat','des_long','client_id','assien_driver','date','time',"price","seat","distance","client_request_number","driver_request_number"]

class Customerserializer(serializers.ModelSerializer):
    class Meta:
        model=Customer
        fields=['id','username','email','password','phone','carplate','carmodel','image1','image2','image3',"balance","trips_as_client","trips_as_captain","Profile","expo_token","point"]

# save client carpool 
class Carpoolreqserializer(serializers.ModelSerializer):
    class Meta:
        model=Carpool_request
        fields=['id','lat','long','des_lat','des_long','client_id','assien_driver','date','time',"price","seat","distance",'status',"client_request_number","driver_request_number"]
# Final Carpool 
class Finalcarpoolserializer(serializers.ModelSerializer):
    class Meta:
        model=Final_Carpool
        fields=['id','lat','long','des_lat','des_long','client_id','assien_driver','date','time',"price","seat","distance",'seat','sheuler_number','picksheuler_number','current_long','current_lat','status',"client_request_number","driver_request_number"]

# save driver carpool
class DCarpoolreqserializer(serializers.ModelSerializer):
    class Meta:
        model=DCarpool_request
        fields=['id','lat','long','des_lat','des_long','client_id','assien_driver','date','time',"price","seat","distance",'status',"client_request_number","driver_request_number","range"]


class Voucherserializer(serializers.ModelSerializer):
    class Meta:
        model=Voucher
        fields=['id','price','voucher','product','driver_id','Qr_code']
class HDrivercarpoolserializer(serializers.ModelSerializer):
    class Meta:
        model=History_Driver
        fields=['id','lat','long','des_lat','des_long','client_id','assien_driver','date','time',"price","seat","distance",'seat','sheuler_number','picksheuler_number','current_long','current_lat']
class HClientcarpoolserializer(serializers.ModelSerializer):
    class Meta:
        model=History_Client
        fields=['id','lat','long','des_lat','des_long','client_id','assien_driver','date','time',"price","seat","distance",'seat','sheuler_number','picksheuler_number','current_long','current_lat']
