package tensor

import "fmt"

func Fused0020SqrtAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	return r, nil
}
func Fused0020SqrtAddName() string { return "sqrt_add" }
