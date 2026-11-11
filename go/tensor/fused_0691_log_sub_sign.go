package tensor

import "fmt"

func Fused0691LogSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Sign()
	return r, nil
}
func Fused0691LogSubSignName() string { return "log_sub_sign" }
