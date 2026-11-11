package tensor

import "fmt"

func Fused0010TanhAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	return r, nil
}
func Fused0010TanhAddName() string { return "tanh_add" }
