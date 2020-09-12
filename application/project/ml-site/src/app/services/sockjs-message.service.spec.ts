import { TestBed } from '@angular/core/testing';

import { SockjsMessageService } from './sockjs-message.service';

describe('SockjsMessageService', () => {
  let service: SockjsMessageService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SockjsMessageService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
