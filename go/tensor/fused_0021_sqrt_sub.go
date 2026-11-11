package tensor

import "fmt"

func Fused0021SqrtSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	return r, nil
}
func Fused0021SqrtSubName() string { return "sqrt_sub" }
