package tensor

import "fmt"

func Fused0690LogAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Sign()
	return r, nil
}
func Fused0690LogAddSignName() string { return "log_add_sign" }
