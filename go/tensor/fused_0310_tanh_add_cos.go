package tensor

import "fmt"

func Fused0310TanhAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Cos()
	return r, nil
}
func Fused0310TanhAddCosName() string { return "tanh_add_cos" }
