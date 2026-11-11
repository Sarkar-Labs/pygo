package tensor

import "fmt"

func Fused0510SqrtAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Sign()
	return r, nil
}
func Fused0510SqrtAddSignName() string { return "sqrt_add_sign" }
