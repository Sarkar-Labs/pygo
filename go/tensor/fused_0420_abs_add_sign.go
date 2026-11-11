package tensor

import "fmt"

func Fused0420AbsAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Sign()
	return r, nil
}
func Fused0420AbsAddSignName() string { return "abs_add_sign" }
