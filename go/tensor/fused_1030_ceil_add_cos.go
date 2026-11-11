package tensor

import "fmt"

func Fused1030CeilAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Cos()
	return r, nil
}
func Fused1030CeilAddCosName() string { return "ceil_add_cos" }
