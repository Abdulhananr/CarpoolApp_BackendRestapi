from asyncio.windows_events import NULL
from sqlite3 import Timestamp
from django.db import models
# from phonenumber_field.modelfields import PhoneNumberField

class Contact(models.Model):
    #------------Customer Suppot -----------
    name = models.CharField(max_length=30)
    email = models.EmailField(max_length=40)
    phone = models.CharField(max_length=14)
    desc = models.TextField()
    def __str__(self):
        return self.email

class Customer(models.Model):

    #-----------User Details -------------------------
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=40,unique=True)
    password=models.CharField(max_length=40)
    phone = models.CharField(max_length=14,unique=True)

    #----------Car Details ------------------------------
    carplate=models.CharField(max_length=14,null=True,blank=True)
    carmodel=models.CharField(max_length=14,null=True,blank=True)
    image1 =models.FileField(upload_to='Media', null=True,blank=True)    
    image2 =models.FileField(upload_to='Media', null=True,blank=True)
    image3 =models.FileField(upload_to='Media', null=True,blank=True)
    
    balance=models.IntegerField(null=True,blank=True)
    trips_as_client=models.IntegerField(null=True,blank=True)
    trips_as_captain=models.IntegerField(null=True,blank=True)
    Profile =models.FileField(upload_to='Media', null=True,blank=True)
    expo_token=models.CharField(max_length=100,null=True,blank=True)
    point=models.IntegerField(null=True,blank=True)
    
    

class Driver_location(models.Model):
    did=models.IntegerField()
    lat=models.FloatField()
    long=models.FloatField()

class Client_location(models.Model):
    cid=models.IntegerField()
    lat=models.FloatField()
    long=models.FloatField()



class Carpool(models.Model):

    lat=models.FloatField()
    long=models.FloatField()
    des_lat=models.FloatField()
    des_long=models.FloatField()
    client_id=models.IntegerField()
    assien_driver=models.IntegerField(null=True,blank=True)
    date=models.CharField(max_length=24,null=True,blank=True)
    time=models.CharField(max_length=24,null=True,blank=True)
    complete=models.CharField(max_length=24,null=True,blank=True)
    price=models.FloatField()
    seat=models.FloatField(null=True,blank=True)
    distance=models.FloatField(null=True,blank=True)
    status=models.CharField(max_length=24,null=True,blank=True)
    client_request_number=models.IntegerField(null=True,blank=True)
    driver_request_number=models.IntegerField(null=True,blank=True)
class Carpool_request(models.Model):

    lat=models.FloatField()
    long=models.FloatField()
    des_lat=models.FloatField()
    des_long=models.FloatField()
    client_id=models.IntegerField()
    assien_driver=models.IntegerField(null=True,blank=True)
    date=models.CharField(max_length=24,null=True,blank=True)
    time=models.CharField(max_length=24,null=True,blank=True)
    price=models.FloatField()
    seat=models.FloatField(null=True,blank=True)
    distance=models.FloatField(null=True,blank=True)
    status=models.CharField(max_length=24,null=True,blank=True)
    client_request_number=models.IntegerField(null=True,blank=True)
    driver_request_number=models.IntegerField(null=True,blank=True)

class Final_Carpool(models.Model):

    lat=models.FloatField()
    long=models.FloatField()
    des_lat=models.FloatField()
    des_long=models.FloatField()
    client_id=models.IntegerField()
    current_lat=models.FloatField(null=True,blank=True)
    current_long=models.FloatField(null=True,blank=True)
    assien_driver=models.IntegerField(null=True,blank=True)
    date=models.CharField(max_length=24,null=True,blank=True)
    time=models.CharField(max_length=24,null=True,blank=True)
    price=models.FloatField()
    seat=models.FloatField(null=True,blank=True)
    distance=models.FloatField(null=True,blank=True)
    seat=models.IntegerField(null=True,blank=True)
    sheuler_number=models.IntegerField(null=True,blank=True)
    picksheuler_number=models.IntegerField(null=True,blank=True)
    status=models.CharField(max_length=24,null=True,blank=True)
    client_request_number=models.IntegerField(null=True,blank=True)
    driver_request_number=models.IntegerField(null=True,blank=True)
class DCarpool_request(models.Model):

    lat=models.FloatField()
    long=models.FloatField()
    des_lat=models.FloatField()
    des_long=models.FloatField()
    client_id=models.IntegerField()
    assien_driver=models.IntegerField(null=True,blank=True)
    date=models.CharField(max_length=24,null=True,blank=True)
    time=models.CharField(max_length=24,null=True,blank=True)
    price=models.FloatField()
    seat=models.FloatField(null=True,blank=True)
    distance=models.FloatField(null=True,blank=True)
    status=models.CharField(max_length=24,null=True,blank=True)
    client_request_number=models.IntegerField(null=True,blank=True)
    driver_request_number=models.IntegerField(null=True,blank=True)
    range=models.IntegerField(null=True,blank=True)

class PrivateRoom(models.Model):
    sender = models.IntegerField(null=True,blank=True)
    reciver=models.IntegerField(null=True,blank=True)
    msg = models.CharField(max_length=500)
    pic = models.CharField(max_length=140)
    name=models.CharField(max_length=500,null=True,blank=True)
    status=models.CharField(max_length=500,null=True,blank=True)
    def __str__(self):
        return self.sender
class Voucher(models.Model):
    price = models.IntegerField(null=True,blank=True)
    voucher= models.IntegerField(null=True,blank=True)
    product=models.IntegerField(null=True,blank=True)
    driver_id=models.IntegerField(null=True,blank=True)
    Qr_code=models.FileField(upload_to='Media', null=True,blank=True)
class History_Driver(models.Model):
    lat=models.FloatField()
    long=models.FloatField()
    des_lat=models.FloatField()
    des_long=models.FloatField()
    client_id=models.IntegerField()
    current_lat=models.FloatField(null=True,blank=True)
    current_long=models.FloatField(null=True,blank=True)
    assien_driver=models.IntegerField(null=True,blank=True)
    date=models.CharField(max_length=24,null=True,blank=True)
    time=models.CharField(max_length=24,null=True,blank=True)
    price=models.FloatField()
    seat=models.FloatField(null=True,blank=True)
    distance=models.FloatField(null=True,blank=True)
    seat=models.IntegerField(null=True,blank=True)
    sheuler_number=models.IntegerField(null=True,blank=True)
    picksheuler_number=models.IntegerField(null=True,blank=True)

    
class History_Client(models.Model):

    lat=models.FloatField()
    long=models.FloatField()
    des_lat=models.FloatField()
    des_long=models.FloatField()
    client_id=models.IntegerField()
    current_lat=models.FloatField(null=True,blank=True)
    current_long=models.FloatField(null=True,blank=True)
    assien_driver=models.IntegerField(null=True,blank=True)
    date=models.CharField(max_length=24,null=True,blank=True)
    time=models.CharField(max_length=24,null=True,blank=True)
    price=models.FloatField()
    seat=models.FloatField(null=True,blank=True)
    distance=models.FloatField(null=True,blank=True)
    seat=models.IntegerField(null=True,blank=True)
    sheuler_number=models.IntegerField(null=True,blank=True)
    picksheuler_number=models.IntegerField(null=True,blank=True)

    
class Carrypic(models.Model):
    pic=models.FileField(upload_to='Media', null=True,blank=True)
    