import { Injectable } from '@angular/core';
import { Filter } from '../../models/filter';
import { Observable, of } from 'rxjs';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { BehaviorSubject } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class FilterService {

  private filtersUrls = [environment.filtersUrl, environment.playgroundUrl];
  private filtersSubject = new BehaviorSubject<Filter[]>([]);

  constructor(private http: HttpClient) {
    for (var filtersUrl of this.filtersUrls) {
      this.http.get<Filter[]>(filtersUrl).subscribe(
        (filters)=>{
          let curFilters = this.filtersSubject.getValue().concat(filters)
          this.filtersSubject.next(curFilters);
        }
      );
    }
  }

  getFilters(): Observable<Filter[]> {
    return this.filtersSubject.asObservable();
  }
}
