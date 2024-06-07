from django.contrib import admin
from django.urls import path,include
from rest_framework.routers import DefaultRouter
from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls import url
from django.views.static import serve
from .views import Contactviewset,Login,Clientviewset,Driverviewset,Carpoolviewset,FinalCarpoolviewset
from .views import Customerviewset,Otp,Check,Checkemail,Checkusername,GALog,Carpool_requestviewset,DCarpoolviewset
from .views import UpdateBalance,ChatCheck,PrivateRoomviewset,Voucherviewset,DeatilsofProdut
from .views import Shedulder,DropofSheduler,GetDetailsofSheduler,DoneCarpool,CreateVoucher,VoucherCheck
from .views import HDriverCarpoolviewset,HClientCarpoolviewset,ClientRequestnumber,DriverRequestnumber,FinalCarpoolsdata,Carrypic_requestviewset
from .views import Changepicture1,Changepicture2,Changepicture3
from .views import ChangeUsercarplate,ChangeUsername,ChangeUserPassword,ChangeUsercarmodel,Changepicture

router=DefaultRouter()
router.register('Contact',Contactviewset,basename='Contact')
router.register('Customer',Customerviewset,basename='Customer')
router.register('Clocation',Clientviewset,basename='Clocation')
router.register('Dlocation',Driverviewset,basename='Dlocation')
router.register('Carpool',Carpoolviewset,basename='Carpool')
router.register('Carpoolre',Carpool_requestviewset,basename='Carpoolre')
router.register('DCarpoolre',DCarpoolviewset,basename='DCarpoolre')
router.register('FinalCarpool',FinalCarpoolviewset,basename='FinalCarpoolviewset')
router.register('msg',PrivateRoomviewset,basename='msg')
router.register('Voucher',Voucherviewset,basename='Voucher')
router.register('DriverHistory',HDriverCarpoolviewset,basename='DriverHistory')
router.register('ClientHistory',HClientCarpoolviewset,basename='ClientHistory')

router.register('Carrypic',Carrypic_requestviewset,basename='Carrypic')

# http://192.168.1.16/api/Otp/+923090389688/1122
urlpatterns = [
    path('api/',include(router.urls)),
    path('api/Login/<str:data>/<str:password>', Login.as_view(), name='Login'),
    path('api/Otp/<str:data>/<int:otp>', Otp.as_view(), name='Otp'),
    path('api/ChatCheck/<str:data>', ChatCheck.as_view(), name='Check'),
    path('api/Add/<int:data>/<int:otp>', UpdateBalance.as_view(), name='Otp'),
    path('api/Check/<str:data>', Check.as_view(), name='Check'),
    path('api/Checkusername/<str:data>', Checkusername.as_view(), name='Checkusername'),
    path('api/Checkemail/<str:data>', Checkemail.as_view(), name='Checkemail'),
    path('api/CheckGa', GALog.as_view(), name='CheckGa'),
    path('api/Shedulder', Shedulder.as_view(), name='Shedulder'),
    path('api/Sheduldedrop', DropofSheduler.as_view(), name='Shedulderdrop'),
    path('api/CheckPickDrop/<int:driver_id>/<int:id_of>', GetDetailsofSheduler.as_view(), name='CheckPickDrop'),
    path('api/DoneCarpool/<int:driver_id>/<int:id_of>', DoneCarpool.as_view(), name='DoneCarpool'),
    path('api/CreateVoucher/<int:price>/<int:clientid>', CreateVoucher.as_view(), name='DoneCarpool'),
    path('api/VoucherCheck/<int:number>/<int:price>', VoucherCheck.as_view(), name='VoucherCheck'),
    path('api/Drivernum/<int:driver_id>', DriverRequestnumber.as_view(), name='Drivernum'),
    path('api/Clientnum/<int:driver_id>', ClientRequestnumber.as_view(), name='Clientnum'),
    path('api/Fetchdata/<int:driver_id>', FinalCarpoolsdata.as_view(), name='Fetchdata'),
    path('api/DeatilsofProdut/<int:driver_id>',DeatilsofProdut.as_view(), name='DeatilsofProdut'),
    path('api/ChangeUsername/<int:driver_id>/<str:name>',ChangeUsername.as_view(), name='ChangeUsername'),
    path('api/ChangeUserPassword/<int:driver_id>/<str:password>',ChangeUserPassword.as_view(), name='ChangeUserPassword'),
    path('api/ChangeUsercarmodel/<int:driver_id>/<str:carmodel>',ChangeUsercarmodel.as_view(), name='ChangeUsercarmodel'),
    path('api/ChangeUsercarplate/<int:driver_id>/<str:carplate>',ChangeUsercarplate.as_view(), name='ChangeUsercarplate'),
    path('api/Changepicture/<int:driver_id>/<int:pic_id>',Changepicture.as_view(), name='Changepicture'),
    path('api/Changepicture1/<int:driver_id>/<int:pic_id>',Changepicture1.as_view(), name='Changepicture1'),
    path('api/Changepicture2/<int:driver_id>/<int:pic_id>',Changepicture2.as_view(), name='Changepicture2'),
    path('api/Changepicture3/<int:driver_id>/<int:pic_id>',Changepicture3.as_view(), name='Changepicture3'),



]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
