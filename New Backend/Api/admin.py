from django.contrib import admin

# Register your models here.
from .models import Contact, Carpool,Voucher
from .models import Customer, Client_location, Driver_location, DCarpool_request, Carpool_request, Final_Carpool

admin.site.register(Carpool_request)
admin.site.register(DCarpool_request)
admin.site.register(Contact)
admin.site.register(Carpool)
admin.site.register(Customer)
admin.site.register(Client_location)
admin.site.register(Voucher)

admin.site.register(Driver_location)
admin.site.register(Final_Carpool)
admin.site.site_header = "Carpool App"
admin.site.site_title = "Carpool App"
admin.site.index_title = "Welcome to Carpool App Admin Portal"