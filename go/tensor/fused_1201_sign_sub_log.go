package tensor

import "fmt"

func Fused1201SignSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Log()
	return r, nil
}
func Fused1201SignSubLogName() string { return "sign_sub_log" }
