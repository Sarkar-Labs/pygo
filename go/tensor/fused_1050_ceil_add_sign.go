package tensor

import "fmt"

func Fused1050CeilAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Sign()
	return r, nil
}
func Fused1050CeilAddSignName() string { return "ceil_add_sign" }
