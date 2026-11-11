package tensor

import "fmt"

func Fused1200SignAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Log()
	return r, nil
}
func Fused1200SignAddLogName() string { return "sign_add_log" }
